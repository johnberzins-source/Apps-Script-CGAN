/**
 * Tiny conditional GAN (cGAN) in Google Apps Script
 * - Toy MLP-based generator and discriminator
 * - Image size: 16x16 (RGB 24-bit BMP)
 * - No external libraries or APIs
 *
 * Usage:
 *  - trainOnBase64(base64Array, conditionArray, epochs, batchSize, checkpointEvery)
 *  - generate(condition, seed)
 *  - saveModel(name)
 *  - loadModel(name)
 *
 * NOTE: base64 images must be 24-bit BMP, 16x16. The generator outputs 24-bit BMP base64.
 */

// -------------------- Config --------------------
const IMG_W = 16;
const IMG_H = 16;
const IMG_C = 3; // RGB
const IMG_PIXELS = IMG_W * IMG_H * IMG_C;
const LATENT_DIM = 64;
const CONDITION_DIM = 8; // small condition vector size
const G_HIDDEN = 256;
const D_HIDDEN = 256;
const LEARNING_RATE = 0.0002;
const BETA1 = 0.5;
const EPS = 1e-8;

// -------------------- Utilities --------------------
function randn(seed) {
  // simple seeded normal generator (Box-Muller)
  let s = seed || Math.floor(Math.random() * 1e9);
  return function() {
    s = (s * 1664525 + 1013904223) >>> 0;
    const u1 = (s & 0xffff) / 65536 || 1e-6;
    s = (s * 1664525 + 1013904223) >>> 0;
    const u2 = (s & 0xffff) / 65536 || 1e-6;
    const r = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return r;
  };
}

function zeros(n) { return new Float32Array(n); }
function randf(n, scale, rng) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (rng ? rng() : (Math.random()*2-1)) * (scale || 1);
  return out;
}

function flatten2d(arr2d) {
  const n = arr2d.length, m = arr2d[0].length;
  const out = new Float32Array(n*m);
  for (let i=0;i<n;i++) for (let j=0;j<m;j++) out[i*m+j]=arr2d[i][j];
  return out;
}

// -------------------- Simple linear layer with manual grads --------------------
function Linear(inDim, outDim, name, rng) {
  this.inDim = inDim;
  this.outDim = outDim;
  this.name = name || '';
  const scale = Math.sqrt(2 / inDim);
  this.W = randf(inDim * outDim, scale, rng);
  this.b = zeros(outDim);

  // Adam state
  this.mW = zeros(this.W.length); this.vW = zeros(this.W.length);
  this.mb = zeros(this.b.length); this.vb = zeros(this.b.length);

  // forward cache
  this.lastInput = null;
}
Linear.prototype.forward = function(x /* Float32Array length inDim */) {
  // x is 1D vector
  const out = new Float32Array(this.outDim);
  for (let o=0;o<this.outDim;o++) {
    let s = this.b[o];
    for (let i=0;i<this.inDim;i++) s += this.W[o*this.inDim + i] * x[i];
    out[o] = s;
  }
  this.lastInput = x;
  return out;
};
Linear.prototype.backward = function(gradOut /* Float32Array length outDim */) {
  // returns gradInput (length inDim) and accumulates grads for W and b
  const gradIn = new Float32Array(this.inDim);
  // grad b
  this.grad_b = gradOut.slice();
  // grad W
  this.grad_W = new Float32Array(this.W.length);
  for (let o=0;o<this.outDim;o++) {
    for (let i=0;i<this.inDim;i++) {
      const idx = o*this.inDim + i;
      this.grad_W[idx] = gradOut[o] * this.lastInput[i];
      gradIn[i] += this.W[idx] * gradOut[o];
    }
  }
  return gradIn;
};
Linear.prototype.applyAdam = function(lr, beta1, beta2, eps, t) {
  // update W and b using stored grads
  const b1 = beta1, b2 = beta2;
  for (let i=0;i<this.W.length;i++) {
    this.mW[i] = b1 * this.mW[i] + (1 - b1) * this.grad_W[i];
    this.vW[i] = b2 * this.vW[i] + (1 - b2) * (this.grad_W[i] * this.grad_W[i]);
    const mHat = this.mW[i] / (1 - Math.pow(b1, t));
    const vHat = this.vW[i] / (1 - Math.pow(b2, t));
    this.W[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
  }
  for (let i=0;i<this.b.length;i++) {
    this.mb[i] = b1 * this.mb[i] + (1 - b1) * this.grad_b[i];
    this.vb[i] = b2 * this.vb[i] + (1 - b2) * (this.grad_b[i] * this.grad_b[i]);
    const mHat = this.mb[i] / (1 - Math.pow(b1, t));
    const vHat = this.vb[i] / (1 - Math.pow(b2, t));
    this.b[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
  }
};

// -------------------- Activations --------------------
function relu(x) {
  const y = new Float32Array(x.length);
  for (let i=0;i<x.length;i++) y[i] = Math.max(0, x[i]);
  return y;
}
function reluBackward(x, gradOut) {
  const g = new Float32Array(x.length);
  for (let i=0;i<x.length;i++) g[i] = (x[i] > 0) ? gradOut[i] : 0;
  return g;
}
function tanhAct(x) {
  const y = new Float32Array(x.length);
  for (let i=0;i<x.length;i++) y[i] = Math.tanh(x[i]);
  return y;
}
function tanhBackward(y, gradOut) {
  const g = new Float32Array(y.length);
  for (let i=0;i<y.length;i++) g[i] = (1 - y[i]*y[i]) * gradOut[i];
  return g;
}
function sigmoid(x) {
  const y = new Float32Array(x.length);
  for (let i=0;i<x.length;i++) y[i] = 1 / (1 + Math.exp(-x[i]));
  return y;
}
function sigmoidBackward(y, gradOut) {
  const g = new Float32Array(y.length);
  for (let i=0;i<y.length;i++) g[i] = y[i] * (1 - y[i]) * gradOut[i];
  return g;
}

// -------------------- Generator --------------------
function Generator(rng) {
  rng = rng || randn();
  // input: latent (LATENT_DIM) + condition (CONDITION_DIM)
  this.fc1 = new Linear(LATENT_DIM + CONDITION_DIM, G_HIDDEN, 'g_fc1', rng);
  this.fc2 = new Linear(G_HIDDEN, G_HIDDEN, 'g_fc2', rng);
  this.fc3 = new Linear(G_HIDDEN, IMG_PIXELS, 'g_fc3', rng);
  this.t = 0;
}
Generator.prototype.forward = function(z, cond) {
  // z: Float32Array LATENT_DIM, cond: Float32Array CONDITION_DIM
  const input = new Float32Array(LATENT_DIM + CONDITION_DIM);
  input.set(z, 0);
  input.set(cond, LATENT_DIM);
  this.x1 = this.fc1.forward(input);
  this.a1 = relu(this.x1);
  this.x2 = this.fc2.forward(this.a1);
  this.a2 = relu(this.x2);
  this.x3 = this.fc3.forward(this.a2);
  // output in [-1,1] via tanh
  this.out = tanhAct(this.x3);
  return this.out; // length IMG_PIXELS, values in [-1,1]
};
Generator.prototype.backward = function(gradOut) {
  // gradOut: gradient w.r.t. output (IMG_PIXELS)
  const g3 = tanhBackward(this.out, gradOut);
  const g2_in = this.fc3.backward(g3);
  const g2 = reluBackward(this.x2, g2_in);
  const g1_in = this.fc2.backward(g2);
  const g1 = reluBackward(this.x1, g1_in);
  const g0 = this.fc1.backward(g1);
  // g0 contains grads w.r.t input (latent+cond) but we don't use them here
};
Generator.prototype.applyAdam = function(lr, beta1, beta2, eps) {
  this.t++;
  this.fc1.applyAdam(lr, beta1, beta2, eps, this.t);
  this.fc2.applyAdam(lr, beta1, beta2, eps, this.t);
  this.fc3.applyAdam(lr, beta1, beta2, eps, this.t);
};

// -------------------- Discriminator --------------------
function Discriminator(rng) {
  rng = rng || randn();
  // input: image (IMG_PIXELS) + condition (CONDITION_DIM)
  this.fc1 = new Linear(IMG_PIXELS + CONDITION_DIM, D_HIDDEN, 'd_fc1', rng);
  this.fc2 = new Linear(D_HIDDEN, D_HIDDEN, 'd_fc2', rng);
  this.fc3 = new Linear(D_HIDDEN, 1, 'd_fc3', rng);
  this.t = 0;
}
Discriminator.prototype.forward = function(img, cond) {
  // img: Float32Array IMG_PIXELS (values in [-1,1])
  const input = new Float32Array(IMG_PIXELS + CONDITION_DIM);
  input.set(img, 0);
  input.set(cond, IMG_PIXELS);
  this.x1 = this.fc1.forward(input);
  this.a1 = relu(this.x1);
  this.x2 = this.fc2.forward(this.a1);
  this.a2 = relu(this.x2);
  this.x3 = this.fc3.forward(this.a2);
  // output raw logit
  this.out = this.x3; // length 1
  return this.out[0];
};
Discriminator.prototype.backward = function(gradOutScalar) {
  const gradOut = new Float32Array(1); gradOut[0] = gradOutScalar;
  const g3 = gradOut; // linear
  const g2_in = this.fc3.backward(g3);
  const g2 = reluBackward(this.x2, g2_in);
  const g1_in = this.fc2.backward(g2);
  const g1 = reluBackward(this.x1, g1_in);
  const g0 = this.fc1.backward(g1);
  // g0 contains grads w.r.t input (img+cond)
  // return gradient w.r.t image portion
  const gradImg = g0.subarray(0, IMG_PIXELS);
  return gradImg;
};
Discriminator.prototype.applyAdam = function(lr, beta1, beta2, eps) {
  this.t++;
  this.fc1.applyAdam(lr, beta1, beta2, eps, this.t);
  this.fc2.applyAdam(lr, beta1, beta2, eps, this.t);
  this.fc3.applyAdam(lr, beta1, beta2, eps, this.t);
};

// -------------------- Loss helpers --------------------
function bceLossWithLogits(logit, target) {
  // logit: scalar, target: 0 or 1
  // stable BCE with logits: max(x,0) - x*y + log(1+exp(-abs(x)))
  const x = logit;
  const y = target;
  const m = Math.max(x, 0);
  const loss = m - x*y + Math.log(1 + Math.exp(-Math.abs(x)));
  // gradient wrt logit:
  const sigmoid_x = 1 / (1 + Math.exp(-x));
  const grad = sigmoid_x - y;
  return {loss: loss, grad: grad};
}

// -------------------- Model init --------------------
let G = null, D = null;
function initModels(seed) {
  const rng = randn(seed || Math.floor(Math.random()*1e9));
  G = new Generator(rng);
  D = new Discriminator(rng);
}

// -------------------- BMP encode/decode (24-bit) --------------------
function bmpFromRGBBytes(width, height, rgbBytes) {
  // rgbBytes: Uint8Array length width*height*3, row-major R G B
  // BMP stores rows bottom-up and BGR order, padded to 4 bytes per row
  const rowSize = Math.ceil((3 * width) / 4) * 4;
  const pixelArraySize = rowSize * height;
  const fileSize = 54 + pixelArraySize;
  const buffer = [];
  function pushUint8(v) { buffer.push(v & 0xff); }
  function pushUint32LE(v) {
    pushUint8(v & 0xff); pushUint8((v>>8)&0xff); pushUint8((v>>16)&0xff); pushUint8((v>>24)&0xff);
  }
  // BMP header
  buffer.push(0x42, 0x4D); // 'BM'
  pushUint32LE(fileSize);
  pushUint16LE(0); pushUint16LE(0);
  pushUint32LE(54);
  // DIB header (BITMAPINFOHEADER)
  pushUint32LE(40); // header size
  pushUint32LE(width);
  pushUint32LE(height);
  pushUint16LE(1); // planes
  pushUint16LE(24); // bits per pixel
  pushUint32LE(0); // compression
  pushUint32LE(pixelArraySize);
  pushUint32LE(2835); pushUint32LE(2835); // ppm
  pushUint32LE(0); pushUint32LE(0);

  // pixel data bottom-up
  for (let y = height - 1; y >= 0; y--) {
    let rowBytes = 0;
    for (let x = 0; x < width; x++) {
      const idx = (y*width + x) * 3;
      const r = rgbBytes[idx], g = rgbBytes[idx+1], b = rgbBytes[idx+2];
      // BMP uses B G R
      buffer.push(b, g, r);
      rowBytes += 3;
    }
    // padding
    while (rowBytes % 4 !== 0) { buffer.push(0); rowBytes++; }
  }

  // helper to push uint16
  function pushUint16LE(v) { buffer.push(v & 0xff); buffer.push((v>>8)&0xff); }

  // base64 encode
  const bytes = new Uint8Array(buffer);
  const base64 = Utilities.base64Encode(bytes);
  return 'data:image/bmp;base64,' + base64;
}

function rgbBytesFromBmpBase64(dataUrlOrBase64) {
  // Accept either "data:image/bmp;base64,..." or raw base64
  const base64 = dataUrlOrBase64.indexOf('base64,') >= 0 ? dataUrlOrBase64.split('base64,')[1] : dataUrlOrBase64;
  const bytes = Utilities.base64Decode(base64);
  // parse BMP header minimally (assumes 24-bit uncompressed)
  const dv = bytes; // Uint8Array
  function readUInt32LE(off) { return dv[off] | (dv[off+1]<<8) | (dv[off+2]<<16) | (dv[off+3]<<24); }
  function readUInt16LE(off) { return dv[off] | (dv[off+1]<<8); }
  const bfType = String.fromCharCode(dv[0], dv[1]);
  if (bfType !== 'BM') throw new Error('Not a BMP');
  const dataOffset = readUInt32LE(10);
  const dibSize = readUInt32LE(14);
  const width = readUInt32LE(18);
  const height = readUInt32LE(22);
  const bpp = readUInt16LE(28);
  if (bpp !== 24) throw new Error('Only 24-bit BMP supported');
  // read pixel data bottom-up
  const rowSize = Math.ceil((3 * width) / 4) * 4;
  const rgb = new Uint8Array(width * height * 3);
  let p = dataOffset;
  for (let y = height - 1; y >= 0; y--) {
    let x = 0;
    for (; x < width; x++) {
      const b = dv[p++], g = dv[p++], r = dv[p++];
      const idx = (y*width + x) * 3;
      rgb[idx] = r; rgb[idx+1] = g; rgb[idx+2] = b;
    }
    // skip padding
    while ((p - dataOffset) % rowSize !== 0) p++;
  }
  return {width: width, height: height, rgb: rgb};
}

// -------------------- Helpers: normalize / denormalize --------------------
function normalizePixelsFromRgbBytes(rgbBytes) {
  // rgbBytes Uint8Array -> Float32Array in [-1,1]
  const out = new Float32Array(rgbBytes.length);
  for (let i=0;i<rgbBytes.length;i++) out[i] = (rgbBytes[i] / 127.5) - 1;
  return out;
}
function denormalizeToRgbBytes(floatArr) {
  // floatArr in [-1,1] -> Uint8Array 0-255
  const out = new Uint8Array(floatArr.length);
  for (let i=0;i<floatArr.length;i++) {
    let v = Math.round((floatArr[i] + 1) * 127.5);
    if (v < 0) v = 0; if (v > 255) v = 255;
    out[i] = v;
  }
  return out;
}

// -------------------- Public API --------------------
/**
 * Train on base64 BMP images.
 * base64Array: array of base64 BMP strings (24-bit, 16x16)
 * conditionArray: array of Float32Array or arrays length CONDITION_DIM (values in [-1,1] recommended)
 * epochs: number
 * batchSize: number
 * checkpointEvery: save model every N batches (optional)
 */
function trainOnBase64(base64Array, conditionArray, epochs, batchSize, checkpointEvery) {
  if (!G || !D) initModels();
  epochs = epochs || 1;
  batchSize = batchSize || 1;
  checkpointEvery = checkpointEvery || 0;

  // decode dataset
  const dataset = [];
  for (let i=0;i<base64Array.length;i++) {
    const obj = rgbBytesFromBmpBase64(base64Array[i]);
    if (obj.width !== IMG_W || obj.height !== IMG_H) throw new Error('Image must be 16x16');
    const norm = normalizePixelsFromRgbBytes(obj.rgb);
    const cond = conditionArray[i];
    const condArr = (cond instanceof Float32Array) ? cond : new Float32Array(cond);
    dataset.push({img: norm, cond: condArr});
  }

  const rng = randn();
  let globalStep = 0;
  for (let ep=0; ep<epochs; ep++) {
    // simple epoch loop (no shuffle for simplicity)
    for (let i=0;i<dataset.length;i+=batchSize) {
      const batch = dataset.slice(i, i+batchSize);
      // For each sample in batch, do D update then G update
      for (let s=0;s<batch.length;s++) {
        const sample = batch[s];
        // 1) Train Discriminator on real
        const realImg = sample.img;
        const cond = sample.cond;
        const dRealLogit = D.forward(realImg, cond);
        const realLoss = bceLossWithLogits(dRealLogit, 1);
        // grad wrt logit
        const gradLogitReal = realLoss.grad;
        // backprop through D
        const gradImgFromReal = D.backward(gradLogitReal);
        // We do not update generator here

        // 2) Train Discriminator on fake
        // sample z
        const z = new Float32Array(LATENT_DIM);
        for (let k=0;k<LATENT_DIM;k++) z[k] = rng();
        const fakeOut = G.forward(z, cond); // [-1,1]
        const dFakeLogit = D.forward(fakeOut, cond);
        const fakeLoss = bceLossWithLogits(dFakeLogit, 0);
        const gradLogitFake = fakeLoss.grad;
        const gradImgFromFake = D.backward(gradLogitFake);

        // Combine grads for D updates: grads are stored in layer objects; apply Adam
        D.applyAdam(LEARNING_RATE, BETA1, 0.999, EPS);

        // 3) Train Generator: want D(G(z)) -> 1
        // forward again (fresh z)
        const z2 = new Float32Array(LATENT_DIM);
        for (let k=0;k<LATENT_DIM;k++) z2[k] = rng();
        const fakeOut2 = G.forward(z2, cond);
        const dFakeLogit2 = D.forward(fakeOut2, cond);
        const genLoss = bceLossWithLogits(dFakeLogit2, 1);
        const gradLogitGen = genLoss.grad;
        // backprop through D to get grad wrt fake image
        const gradImgForG = D.backward(gradLogitGen);
        // backprop through G
        G.backward(gradImgForG);
        // apply Adam to G
        G.applyAdam(LEARNING_RATE, BETA1, 0.999, EPS);

        globalStep++;
        if (checkpointEvery > 0 && (globalStep % checkpointEvery) === 0) {
          saveModel('cgan_checkpoint');
        }
      } // end batch samples
    } // end dataset loop
  } // end epochs
  return {status: 'done', epochs: epochs};
}

/**
 * Generate an image for a given condition.
 * condition: array or Float32Array length CONDITION_DIM
 * seed: optional integer seed
 * returns: data URL base64 BMP string
 */
function generate(condition, seed) {
  if (!G) initModels();
  const condArr = (condition instanceof Float32Array) ? condition : new Float32Array(condition);
  const rng = randn(seed || Math.floor(Math.random()*1e9));
  const z = new Float32Array(LATENT_DIM);
  for (let i=0;i<LATENT_DIM;i++) z[i] = rng();
  const out = G.forward(z, condArr); // [-1,1]
  const rgb = denormalizeToRgbBytes(out);
  const dataUrl = bmpFromRGBBytes(IMG_W, IMG_H, rgb);
  return dataUrl;
}

/**
 * Save model weights to Drive as JSON (small models only)
 */
function saveModel(name) {
  if (!G || !D) throw new Error('Models not initialized');
  const payload = {
    G: {
      fc1: {W: Array.from(G.fc1.W), b: Array.from(G.fc1.b)},
      fc2: {W: Array.from(G.fc2.W), b: Array.from(G.fc2.b)},
      fc3: {W: Array.from(G.fc3.W), b: Array.from(G.fc3.b)}
    },
    D: {
      fc1: {W: Array.from(D.fc1.W), b: Array.from(D.fc1.b)},
      fc2: {W: Array.from(D.fc2.W), b: Array.from(D.fc2.b)},
      fc3: {W: Array.from(D.fc3.W), b: Array.from(D.fc3.b)}
    }
  };
  const fileName = (name || 'cgan_model') + '.json';
  const file = DriveApp.createFile(fileName, JSON.stringify(payload));
  return {status: 'saved', fileId: file.getId()};
}

/**
 * Load model weights from Drive JSON file (file name or id)
 */
function loadModel(nameOrId) {
  let file = null;
  try {
    file = DriveApp.getFileById(nameOrId);
  } catch (e) {
    // try by name
    const files = DriveApp.getFilesByName(nameOrId + '.json');
    if (files.hasNext()) file = files.next();
  }
  if (!file) throw new Error('Model file not found');
  const content = file.getBlob().getDataAsString();
  const payload = JSON.parse(content);
  initModels();
  // assign weights
  G.fc1.W.set(payload.G.fc1.W); G.fc1.b.set(payload.G.fc1.b);
  G.fc2.W.set(payload.G.fc2.W); G.fc2.b.set(payload.G.fc2.b);
  G.fc3.W.set(payload.G.fc3.W); G.fc3.b.set(payload.G.fc3.b);
  D.fc1.W.set(payload.D.fc1.W); D.fc1.b.set(payload.D.fc1.b);
  D.fc2.W.set(payload.D.fc2.W); D.fc2.b.set(payload.D.fc2.b);
  D.fc3.W.set(payload.D.fc3.W); D.fc3.b.set(payload.D.fc3.b);
  return {status: 'loaded'};
}

// -------------------- Example helper to create a grayscale BMP base64 from a 16x16 array --------------------
function exampleMakeBase64FromGrayArray(grayArray) {
  // grayArray length 256 values 0-255
  const rgb = new Uint8Array(IMG_PIXELS);
  for (let i=0;i<IMG_W*IMG_H;i++) {
    const v = grayArray[i];
    rgb[i*3] = v; rgb[i*3+1] = v; rgb[i*3+2] = v;
  }
  return bmpFromRGBBytes(IMG_W, IMG_H, rgb);
}


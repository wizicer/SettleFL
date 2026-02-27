const tf =
  process.env.TFJS_BACKEND === "GPU"
    ? require("@tensorflow/tfjs-node-gpu")
    : process.env.TFJS_BACKEND === "CPU"
    ? require("@tensorflow/tfjs-node")
    : require("@tensorflow/tfjs");

function createLeNet5Model(learningRate = 0.01) {
  const model = tf.sequential();

  // First Convolutional Layer
  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 6,
      activation: "relu",
      padding: "same",
    })
  );

  // First Max Pooling Layer
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  );

  // Second Convolutional Layer
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      activation: "relu",
      padding: "valid",
    })
  );

  // Second Max Pooling Layer
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  );

  // Flatten layer
  model.add(tf.layers.flatten());

  // First Dense Layer
  model.add(
    tf.layers.dense({
      units: 120,
      activation: "relu",
    })
  );

  // Second Dense Layer
  model.add(
    tf.layers.dense({
      units: 84,
      activation: "relu",
    })
  );

  // Output Layer
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "softmax",
    })
  );

  // Compile model
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["acc"],
  });

  return model;
}

// ResNet-18 implementation for CIFAR-10
function residualBlock(inputs, filters, strides = 1) {
  let y = tf.layers
    .conv2d({
      filters,
      kernelSize: 3,
      strides,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(inputs);
  y = tf.layers.batchNormalization().apply(y);
  y = tf.layers.reLU().apply(y);

  y = tf.layers
    .conv2d({
      filters,
      kernelSize: 3,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(y);
  y = tf.layers.batchNormalization().apply(y);

  let shortcut = inputs;
  if (strides > 1 || inputs.shape[inputs.shape.length - 1] !== filters) {
    shortcut = tf.layers
      .conv2d({
        filters,
        kernelSize: 1,
        strides,
        kernelInitializer: "heNormal",
      })
      .apply(inputs);
    shortcut = tf.layers.batchNormalization().apply(shortcut);
  }

  y = tf.layers.add().apply([y, shortcut]);
  y = tf.layers.reLU().apply(y);
  return y;
}

function createSimpleCNNModel(learningRate = 0.001) {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      filters: 32,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      units: 128,
      activation: "relu",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["acc"],
  });

  return model;
}

function bottleneckBlock(x, filters, strides = 1) {
  const [f1, f2, f3] = filters;
  let shortcut = x;

  // Main path
  let y = tf.layers
    .conv2d({
      filters: f1,
      kernelSize: 1,
      strides: strides,
      padding: "valid",
      kernelInitializer: "heNormal",
    })
    .apply(x);
  y = tf.layers.batchNormalization().apply(y);
  y = tf.layers.reLU().apply(y);

  y = tf.layers
    .conv2d({
      filters: f2,
      kernelSize: 3,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(y);
  y = tf.layers.batchNormalization().apply(y);
  y = tf.layers.reLU().apply(y);

  y = tf.layers
    .conv2d({
      filters: f3,
      kernelSize: 1,
      padding: "valid",
      kernelInitializer: "heNormal",
    })
    .apply(y);
  y = tf.layers.batchNormalization().apply(y);

  // Shortcut path
  if (strides !== 1 || x.shape[x.shape.length - 1] !== f3) {
    shortcut = tf.layers
      .conv2d({
        filters: f3,
        kernelSize: 1,
        strides: strides,
        padding: "valid",
        kernelInitializer: "heNormal",
      })
      .apply(x);
    shortcut = tf.layers.batchNormalization().apply(shortcut);
  }

  y = tf.layers.add().apply([y, shortcut]);
  y = tf.layers.reLU().apply(y);
  return y;
}

function identityBlock(x, filters, strides = 1) {
  let shortcut = x;

  // Main path
  let y = tf.layers
    .conv2d({
      filters: filters,
      kernelSize: 3,
      strides: strides,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(x);
  y = tf.layers.batchNormalization().apply(y);
  y = tf.layers.reLU().apply(y);

  y = tf.layers
    .conv2d({
      filters: filters,
      kernelSize: 3,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(y);
  y = tf.layers.batchNormalization().apply(y);

  // Shortcut path
  if (strides !== 1 || x.shape[x.shape.length - 1] !== filters) {
    shortcut = tf.layers
      .conv2d({
        filters: filters,
        kernelSize: 1,
        strides: strides,
        padding: "valid",
        kernelInitializer: "heNormal",
      })
      .apply(x);
    shortcut = tf.layers.batchNormalization().apply(shortcut);
  }

  y = tf.layers.add().apply([y, shortcut]);
  y = tf.layers.reLU().apply(y);
  return y;
}

function createResNet20Model(learningRate = 0.01) {
  const inputs = tf.input({ shape: [32, 32, 3] });

  // Initial Conv layer
  let x = tf.layers
    .conv2d({
      filters: 16,
      kernelSize: 3,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(inputs);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.reLU().apply(x);

  // Layer 1 (n=3)
  x = identityBlock(x, 16);
  x = identityBlock(x, 16);
  x = identityBlock(x, 16);

  // Layer 2 (n=3)
  x = identityBlock(x, 32, 2);
  x = identityBlock(x, 32);
  x = identityBlock(x, 32);

  // Layer 3 (n=3)
  x = identityBlock(x, 64, 2);
  x = identityBlock(x, 64);
  x = identityBlock(x, 64);

  // Final layers
  x = tf.layers.globalAveragePooling2d({ dataFormat: "channelsLast" }).apply(x);
  const outputs = tf.layers.dense({ units: 10, activation: "softmax" }).apply(x);

  const model = tf.model({ inputs: inputs, outputs: outputs });

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["acc"],
  });

  return model;
}

function createResNet50Model(learningRate = 0.01) {
  const inputs = tf.input({ shape: [32, 32, 3] });

  // Initial Conv layer (CIFAR-10 adaptation)
  let x = tf.layers
    .conv2d({
      filters: 64,
      kernelSize: 3,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(inputs);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.reLU().apply(x);

  // Stage 1 (conv2_x)
  x = bottleneckBlock(x, [64, 64, 256], 1);
  x = bottleneckBlock(x, [64, 64, 256]);
  x = bottleneckBlock(x, [64, 64, 256]);

  // Stage 2 (conv3_x)
  x = bottleneckBlock(x, [128, 128, 512], 2);
  x = bottleneckBlock(x, [128, 128, 512]);
  x = bottleneckBlock(x, [128, 128, 512]);
  x = bottleneckBlock(x, [128, 128, 512]);

  // Stage 3 (conv4_x)
  x = bottleneckBlock(x, [256, 256, 1024], 2);
  x = bottleneckBlock(x, [256, 256, 1024]);
  x = bottleneckBlock(x, [256, 256, 1024]);
  x = bottleneckBlock(x, [256, 256, 1024]);
  x = bottleneckBlock(x, [256, 256, 1024]);
  x = bottleneckBlock(x, [256, 256, 1024]);

  // Stage 4 (conv5_x)
  x = bottleneckBlock(x, [512, 512, 2048], 2);
  x = bottleneckBlock(x, [512, 512, 2048]);
  x = bottleneckBlock(x, [512, 512, 2048]);

  // Final layers
  x = tf.layers.avgPool2d({ poolSize: 4 }).apply(x);
  x = tf.layers.flatten().apply(x);
  const outputs = tf.layers.dense({ units: 10, activation: "softmax" }).apply(x);

  const model = tf.model({ inputs: inputs, outputs: outputs });

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["acc"],
  });

  return model;
}

function createResNet18Model(learningRate = 0.01) {
  const inputs = tf.input({ shape: [32, 32, 3] });

  let x = tf.layers
    .conv2d({
      filters: 64,
      kernelSize: 3,
      padding: "same",
      kernelInitializer: "heNormal",
    })
    .apply(inputs);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.reLU().apply(x);

  // ResNet-18 layers
  x = residualBlock(x, 64);
  x = residualBlock(x, 64);

  x = residualBlock(x, 128, 2);
  x = residualBlock(x, 128);

  x = residualBlock(x, 256, 2);
  x = residualBlock(x, 256);

  x = residualBlock(x, 512, 2);
  x = residualBlock(x, 512);

  x = tf.layers.globalAveragePooling2d({ dataFormat: "channelsLast" }).apply(x);
  const outputs = tf.layers
    .dense({
      units: 10,
      activation: "softmax",
      kernelInitializer: "heNormal",
    })
    .apply(x);

  const model = tf.model({ inputs, outputs });

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "categoricalCrossentropy",
    metrics: ["acc"],
  });

  return model;
}

/**
 * MLP model for tabular binary classification (e.g. Bank Marketing dataset).
 * Architecture: inputDim -> 128 (relu, dropout 0.3) -> 64 (relu, dropout 0.2) -> 32 (relu) -> 1 (sigmoid)
 * Loss: binaryCrossentropy, Metric: acc
 */
function createMLPModel(inputDim, learningRate = 0.01) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [inputDim],
      units: 128,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(
    tf.layers.dense({
      units: 32,
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );

  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
      kernelInitializer: "glorotUniform",
    })
  );

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "binaryCrossentropy",
    metrics: ["acc"],
  });

  return model;
}

/**
 * Graph Convolutional Layer for GCN.
 * forward(x, adj): output = adj @ (x @ weight) + bias
 */
let gcnLayerCounter = 0;

class GraphConvolution {
  constructor(inFeatures, outFeatures) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.layerId = gcnLayerCounter++;
    const stdv = 1.0 / Math.sqrt(outFeatures);
    this.weight = tf.variable(tf.randomUniform([inFeatures, outFeatures], -stdv, stdv), true, `gcn_weight_${this.layerId}`);
    this.bias = tf.variable(tf.randomUniform([outFeatures], -stdv, stdv), true, `gcn_bias_${this.layerId}`);
  }

  forward(x, adj) {
    const support = tf.matMul(x, this.weight);
    const output = tf.matMul(adj, support);
    return tf.add(output, this.bias);
  }

  getWeights() {
    return { weight: this.weight.arraySync(), bias: this.bias.arraySync() };
  }

  setWeights(weights) {
    this.weight.assign(tf.tensor2d(weights.weight));
    this.bias.assign(tf.tensor1d(weights.bias));
  }

  dispose() {
    this.weight.dispose();
    this.bias.dispose();
  }
}

/**
 * Two-layer GCN for node classification.
 * Architecture: GCConv(inputDim, hiddenDim) -> ReLU -> Dropout -> GCConv(hiddenDim, numClasses)
 * Weights are stored as plain objects (not tf.sequential) for FedAvg compatibility.
 */
class GCNModel {
  constructor(inputDim, hiddenDim, numClasses, dropout = 0.5) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.numClasses = numClasses;
    this.dropoutRate = dropout;
    this.training = true;
    this.gc1 = new GraphConvolution(inputDim, hiddenDim);
    this.gc2 = new GraphConvolution(hiddenDim, numClasses);
  }

  forward(x, adj) {
    let h = this.gc1.forward(x, adj);
    h = tf.relu(h);
    if (this.training && this.dropoutRate > 0) {
      h = tf.dropout(h, this.dropoutRate);
    }
    h = this.gc2.forward(h, adj);
    return h;
  }

  train() {
    this.training = true;
  }
  eval() {
    this.training = false;
  }

  getWeights() {
    return { gc1: this.gc1.getWeights(), gc2: this.gc2.getWeights() };
  }

  setWeights(weights) {
    this.gc1.setWeights(weights.gc1);
    this.gc2.setWeights(weights.gc2);
  }

  getTrainableVariables() {
    return [this.gc1.weight, this.gc1.bias, this.gc2.weight, this.gc2.bias];
  }

  dispose() {
    this.gc1.dispose();
    this.gc2.dispose();
  }
}

/**
 * Factory function for GCN model.
 * @param {number} inputDim - Number of input features per node
 * @param {number} numClasses - Number of output classes
 * @param {number} hiddenDim - Hidden layer size (default 16)
 * @param {number} learningRate
 * @param {number} dropout
 * @returns {GCNModel}
 */
function createGCNModel(inputDim, numClasses, hiddenDim = 16, learningRate = 0.015, dropout = 0.5) {
  return new GCNModel(inputDim, hiddenDim, numClasses, dropout);
}

/**
 * TextCNN model for text classification (sent140 sentiment).
 * Architecture: Embedding -> parallel Conv1D (filter sizes 3,4,5) -> GlobalMaxPool -> Dropout -> Dense(2, softmax)
 * @param {number} vocabSize
 * @param {number} embedDim
 * @param {number} maxLen - sequence length
 * @param {number} learningRate
 */
function createTextCNNModel(vocabSize, embedDim = 128, maxLen = 25, learningRate = 0.001) {
  const numFilters = 100;
  const filterSizes = [3, 4, 5];
  const dropout = 0.5;

  const input = tf.input({ shape: [maxLen], dtype: "int32" });

  const embedding = tf.layers
    .embedding({ inputDim: vocabSize, outputDim: embedDim, inputLength: maxLen, maskZero: false })
    .apply(input);

  const convOutputs = filterSizes.map((fs) => {
    const conv = tf.layers.conv1d({ filters: numFilters, kernelSize: fs, activation: "relu", padding: "valid" }).apply(embedding);
    return tf.layers.globalMaxPooling1d().apply(conv);
  });

  const concatenated = tf.layers.concatenate().apply(convOutputs);
  const dropped = tf.layers.dropout({ rate: dropout }).apply(concatenated);
  const output = tf.layers.dense({ units: 2, activation: "softmax" }).apply(dropped);

  const model = tf.model({ inputs: input, outputs: output });
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

/**
 * FastText model for text classification (sent140 sentiment).
 * Architecture: Embedding -> GlobalAveragePool -> Dense(128, relu) -> Dropout -> Dense(2, softmax)
 * @param {number} vocabSize
 * @param {number} embedDim
 * @param {number} maxLen - sequence length
 * @param {number} learningRate
 */
function createFastTextModel(vocabSize, embedDim = 128, maxLen = 25, learningRate = 0.001) {
  const hiddenDim = 128;
  const dropout = 0.5;

  const input = tf.input({ shape: [maxLen], dtype: "int32" });

  const embedding = tf.layers.embedding({ inputDim: vocabSize, outputDim: embedDim, inputLength: maxLen }).apply(input);

  const avgPool = tf.layers.globalAveragePooling1d().apply(embedding);
  const hidden = tf.layers.dense({ units: hiddenDim, activation: "relu" }).apply(avgPool);
  const dropped = tf.layers.dropout({ rate: dropout }).apply(hidden);
  const output = tf.layers.dense({ units: 2, activation: "softmax" }).apply(dropped);

  const model = tf.model({ inputs: input, outputs: output });
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

module.exports = {
  createLeNet5Model,
  createResNet18Model,
  createSimpleCNNModel,
  createResNet50Model,
  createResNet20Model,
  createMLPModel,
  createGCNModel,
  GCNModel,
  createTextCNNModel,
  createFastTextModel,
};

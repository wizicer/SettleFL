const tf =
  process.env.TFJS_BACKEND === "GPU"
    ? require("@tensorflow/tfjs-node-gpu")
    : process.env.TFJS_BACKEND === "CPU"
    ? require("@tensorflow/tfjs-node")
    : require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");

const {
  createLeNet5Model,
  createResNet18Model,
  createSimpleCNNModel,
  createResNet50Model,
  createResNet20Model,
  createMLPModel,
  createGCNModel,
  createTextCNNModel,
  createFastTextModel,
} = require("./models");
const DataLoader = require("./dataLoader");

class FederatedLearningTrainer {
  constructor(options = {}) {
    this.model = null;
    this.dataDir = options.dataDir || path.join(__dirname, "..", "..", "target", "data");
    this.batchSize = options.localBatchSize || options.batchSize || 32;
    this.epochs = options.epochs || 1;
    this.learningRate = options.learningRate || 0.01;
    this.participantCount = options.participantCount || 3;
    this.rounds = options.rounds || 3;
    this.logLevel = options.logLevel || "info";
    this.useParallelTraining = options.useParallelTraining !== false; // Default to true
    this.dataset = options.dataset || "mnist"; // 'mnist', 'cifar10', 'fashion-mnist', 'bank-marketing', 'cora', or 'sent140'
    this.modelName = options.model || "lenet5"; // 'lenet5', 'resnet18', 'resnet20', 'resnet50', 'simple-cnn', 'mlp', 'gcn', 'textcnn', or 'fasttext'
    this.maxDataSizePerParticipant = options.maxDataSizePerParticipant;

    this.trainingHistory = [];
    this.aggregationHistory = [];
    this.modelWeightsHistory = [];

    // CSV logging system
    this.csvLog = [];
    this.runId = this.generateRunId();
    this.startTime = new Date();
  }

  generateRunId() {
    return `fl_run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  addRoundToHistory(roundResult) {
    this.trainingHistory.push(roundResult);
  }

  logCSV(role, id, address, round, accuracy, loss, time, samples, status = "success", error = null) {
    const entry = {
      run: this.runId,
      role: role, // 'aggregator' or 'participant'
      id: id, // participant ID or 'owner' for aggregator
      address: address, // participant address or 'owner' for aggregator
      round: round,
      accuracy: accuracy,
      loss: loss,
      time: time, // training time in seconds or evaluation time
      samples: samples, // training samples or test samples
      timestamp: new Date().toISOString(),
      status: status,
      error: error || "",
      // Additional useful parameters
      epochs: this.epochs,
      batch_size: this.batchSize,
      learning_rate: this.learningRate,
      model_architecture: this.modelName,
      dataset: this.dataset,
      total_rounds: this.trainingHistory.length + 1,
      participant_count: this.participantData ? this.participantData.length : 0,
    };

    this.csvLog.push(entry);
    console.log(`📊 CSV Log: ${role} ${id} - Round ${round} - Acc: ${(accuracy * 100).toFixed(2)}% - Loss: ${loss.toFixed(4)}`);
  }

  getCSVData() {
    return this.csvLog;
  }

  exportCSV() {
    const headers = [
      "run",
      "role",
      "id",
      "address",
      "round",
      "accuracy",
      "loss",
      "time",
      "samples",
      "timestamp",
      "status",
      "error",
      "epochs",
      "batch_size",
      "learning_rate",
      "model_architecture",
      "dataset",
      "total_rounds",
      "participant_count",
    ];
    const csvContent = [
      headers.join(","),
      ...this.csvLog.map((row) =>
        [
          row.run,
          row.role,
          row.id,
          row.address,
          row.round,
          row.accuracy,
          row.loss,
          row.time,
          row.samples,
          row.timestamp,
          row.status,
          row.error,
          row.epochs,
          row.batch_size,
          row.learning_rate,
          row.model_architecture,
          row.dataset,
          row.total_rounds,
          row.participant_count,
        ].join(",")
      ),
    ].join("\n");

    return csvContent;
  }

  log(level, message, data = null) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;

    if (this.logLevel === "none") {
      return;
    }

    if (this.logLevel === "debug" || level === "error" || level === "warn" || level === "info") {
      console.log(logMessage);
      if (data) {
        console.log(JSON.stringify(data, null, 2));
      }
    }
  }

  async initialize() {
    this.log("info", "Initializing Federated Learning Trainer");

    // Initialize TensorFlow.js backend
    try {
      await tf.ready();
      this.log("info", "TensorFlow.js backend initialized");
    } catch (error) {
      this.log("warn", "TensorFlow.js backend initialization warning", error);
    }

    // Load and preprocess data
    await this.loadData();

    // Create model based on options
    if (this.modelName === "lenet5") {
      this.model = createLeNet5Model(this.learningRate);
    } else if (this.modelName === "resnet18") {
      this.model = createResNet18Model(this.learningRate);
    } else if (this.modelName === "resnet50") {
      this.model = createResNet50Model(this.learningRate);
    } else if (this.modelName === "resnet20") {
      this.model = createResNet20Model(this.learningRate);
    } else if (this.modelName === "simple-cnn") {
      this.model = createSimpleCNNModel(this.learningRate);
    } else if (this.modelName === "mlp") {
      if (!this.inputDim) {
        throw new Error("inputDim must be set before creating MLP model. Ensure data is loaded first.");
      }
      this.model = createMLPModel(this.inputDim, this.learningRate);
    } else if (this.modelName === "gcn") {
      if (!this.graphData) {
        throw new Error("graphData must be set before creating GCN model. Ensure data is loaded first.");
      }
      this.model = createGCNModel(this.graphData.numFeatures, this.graphData.numClasses, 16, this.learningRate, 0.5);
      // Build shared graph tensors used across all training/evaluation
      this.graphTensors = {
        features: tf.tensor2d(this.graphData.featuresFlat, [this.graphData.numNodes, this.graphData.numFeatures]),
        adj: tf.tensor2d(this.graphData.adjFlat, [this.graphData.numNodes, this.graphData.numNodes]),
        labels: tf.tensor1d(this.graphData.labels, "int32"),
      };
    } else if (this.modelName === "textcnn") {
      if (!this.sent140Meta) {
        throw new Error("sent140Meta must be set before creating TextCNN model. Ensure data is loaded first.");
      }
      this.model = createTextCNNModel(this.sent140Meta.vocabSize, 128, this.sent140Meta.maxLen, this.learningRate);
    } else if (this.modelName === "fasttext") {
      if (!this.sent140Meta) {
        throw new Error("sent140Meta must be set before creating FastText model. Ensure data is loaded first.");
      }
      this.model = createFastTextModel(this.sent140Meta.vocabSize, 128, this.sent140Meta.maxLen, this.learningRate);
    } else {
      throw new Error(`Unsupported model: ${this.modelName}`);
    }

    this.log("info", "Federated Learning Trainer initialized successfully");
  }

  async loadData() {
    const dataLoader = new DataLoader(this.dataDir, this.log.bind(this), this.maxDataSizePerParticipant);
    try {
      const result = await dataLoader.load(this.dataset, this.participantCount);
      this.participantData = result.participantData;
      this.testData = result.testData;
      // For tabular datasets, infer input dimension from the first sample
      if (this.dataset === "bank-marketing" && result.testData.length > 0) {
        this.inputDim = result.testData[0].features.length;
        this.log("info", `Bank Marketing input dimension: ${this.inputDim}`);
      }
      // For graph datasets, store graph metadata
      if (this.dataset === "cora" && result.graphData) {
        this.graphData = result.graphData;
      }
      // For text datasets, store vocab/sequence metadata
      if (this.dataset === "sent140" && result.sent140Meta) {
        this.sent140Meta = result.sent140Meta;
      }
    } catch (error) {
      this.log("error", "Failed to load data", { error: error.message });
      throw error;
    }
  }

  preprocessData(data) {
    this.log("debug", `Preprocessing ${data.length} samples for training`);

    if (!data || data.length === 0) {
      throw new Error("No data provided for preprocessing");
    }

    // Tabular dataset path (bank-marketing)
    if (this.dataset === "bank-marketing") {
      return this.preprocessTabularData(data);
    }

    // Text dataset path (sent140)
    if (this.dataset === "sent140") {
      return this.preprocessSequenceData(data);
    }

    const images = [];
    const labels = [];

    const isCifar = this.dataset === "cifar10";
    const [height, width, channels] = isCifar ? [32, 32, 3] : [28, 28, 1];

    for (const sample of data) {
      if (!sample.pixels || sample.label === undefined) {
        this.log("warn", "Skipping invalid sample:", JSON.stringify(sample).slice(0, 100));
        continue;
      }

      const normalizedPixels = sample.pixels.map((p) => p / 255.0);

      if (isCifar) {
        const r = normalizedPixels.slice(0, 1024);
        const g = normalizedPixels.slice(1024, 2048);
        const b = normalizedPixels.slice(2048, 3072);

        const reshapedImage = [];
        for (let y = 0; y < height; y++) {
          const row = [];
          for (let x = 0; x < width; x++) {
            const index = y * width + x;
            row.push([r[index], g[index], b[index]]);
          }
          reshapedImage.push(row);
        }
        images.push(reshapedImage);
      } else {
        const reshapedImage = [];
        for (let i = 0; i < height; i++) {
          const row = [];
          for (let j = 0; j < width; j++) {
            row.push([normalizedPixels[i * width + j]]);
          }
          reshapedImage.push(row);
        }
        images.push(reshapedImage);
      }

      const oneHotLabel = new Array(10).fill(0);
      oneHotLabel[sample.label] = 1;
      labels.push(oneHotLabel);
    }

    if (images.length === 0) {
      throw new Error("No valid samples after preprocessing");
    }

    this.log("debug", `Preprocessed ${images.length} valid samples`);

    return {
      images: tf.tensor4d(images),
      labels: tf.tensor2d(labels),
    };
  }

  preprocessSequenceData(data) {
    const indices = data.map((s) => s.indices);
    const labels = data.map((s) => s.label);
    return {
      images: tf.tensor2d(indices, [indices.length, indices[0].length], "int32"),
      labels: tf.tensor1d(labels, "float32"),
    };
  }

  preprocessTabularData(data) {
    const validSamples = data.filter((s) => s.features && s.label !== undefined);
    if (validSamples.length === 0) {
      throw new Error("No valid tabular samples after preprocessing");
    }

    const featureMatrix = validSamples.map((s) => s.features);
    const labelVector = validSamples.map((s) => [s.label]);

    this.log("debug", `Preprocessed ${validSamples.length} tabular samples`);

    return {
      images: tf.tensor2d(featureMatrix),
      labels: tf.tensor2d(labelVector),
    };
  }

  async publishModel() {
    this.log("debug", "Publishing global model to participants");

    // Get current model weights
    const globalWeights = this.model.getWeights();

    // Store weights for history
    this.modelWeightsHistory.push({
      round: this.trainingHistory.length + 1,
      weights: globalWeights,
      timestamp: new Date().toISOString(),
    });

    this.log("debug", `Model weights published. Round: ${this.trainingHistory.length + 1}`);

    return globalWeights;
  }

  async localTraining(participantId, participantData, globalWeights, participantAddress) {
    this.log("info", `Starting local training for participant ${participantId + 1}, ${participantData.length} samples`);

    const startTime = Date.now();

    try {
      // Create a copy of the model for local training
      let localModel;
      if (this.modelName === "lenet5") {
        localModel = createLeNet5Model(this.learningRate);
      } else if (this.modelName === "resnet18") {
        localModel = createResNet18Model(this.learningRate);
      } else if (this.modelName === "resnet50") {
        localModel = createResNet50Model(this.learningRate);
      } else if (this.modelName === "resnet20") {
        localModel = createResNet20Model(this.learningRate);
      } else if (this.modelName === "simple-cnn") {
        localModel = createSimpleCNNModel(this.learningRate);
      } else if (this.modelName === "mlp") {
        localModel = createMLPModel(this.inputDim, this.learningRate);
      } else if (this.modelName === "textcnn") {
        localModel = createTextCNNModel(this.sent140Meta.vocabSize, 128, this.sent140Meta.maxLen, this.learningRate);
      } else if (this.modelName === "fasttext") {
        localModel = createFastTextModel(this.sent140Meta.vocabSize, 128, this.sent140Meta.maxLen, this.learningRate);
      } else if (this.modelName === "gcn") {
        return await this.gcnLocalTraining(participantId, participantData, globalWeights, participantAddress, startTime);
      } else {
        throw new Error(`Unsupported model: ${this.modelName}`);
      }

      // Set global weights
      localModel.setWeights(globalWeights);

      // Preprocess participant data
      const processedData = this.preprocessData(participantData);

      this.log("debug", `Participant ${participantId + 1} training on ${participantData.length} samples`);

      // Train locally
      this.log("debug", `Training participant ${participantId + 1} with ${participantData.length} samples`);
      const history = await localModel.fit(processedData.images, processedData.labels, {
        batchSize: this.batchSize,
        epochs: this.epochs,
        verbose: 0,
      });

      this.log("debug", `Training history for participant ${participantId + 1}:`, history.history);

      // Get updated weights and extract data before disposal
      const localWeights = localModel.getWeights();
      const weightData = localWeights.map((weight) => ({
        data: weight.dataSync(),
        shape: weight.shape,
      }));

      // Clean up tensors
      try {
        processedData.images.dispose();
        processedData.labels.dispose();
        localModel.dispose();
      } catch (error) {
        this.log("warn", "Error disposing tensors", error);
      }

      // Extract metrics from history - TensorFlow.js history structure is different
      const loss = Array.isArray(history.history.loss) ? history.history.loss[0] : -1;
      const accuracy = Array.isArray(history.history.acc)
        ? history.history.acc[0]
        : Array.isArray(history.history.accuracy)
        ? history.history.accuracy[0]
        : -1;

      const trainingTime = (Date.now() - startTime) / 1000; // Convert to seconds

      const trainingMetrics = {
        participantId: participantId + 1,
        loss: loss,
        accuracy: accuracy,
        samples: participantData.length,
      };

      // Log CSV data for participant training
      this.logCSV(
        "participant",
        participantId + 1,
        participantAddress,
        this.trainingHistory.length + 1, // Current round
        accuracy,
        loss,
        trainingTime,
        participantData.length,
        "success"
      );

      this.log("debug", `Local training completed for participant ${participantId + 1}`, trainingMetrics);

      return {
        weights: weightData,
        metrics: trainingMetrics,
      };
    } catch (error) {
      const trainingTime = (Date.now() - startTime) / 1000;

      // Log CSV data for failed training
      this.logCSV(
        "participant",
        participantId + 1,
        participantAddress,
        this.trainingHistory.length + 1,
        -1,
        -1,
        trainingTime,
        participantData.length,
        "error",
        error.message
      );

      this.log("error", `Error in local training for participant ${participantId + 1}`, error);
      throw error;
    }
  }

  async uploadWeights(participantId, localTrainingResult) {
    this.log("debug", `Participant ${participantId + 1} uploading weights`);

    const uploadResult = {
      participantId: participantId + 1,
      weights: localTrainingResult.weights, // Already extracted data
      metrics: localTrainingResult.metrics,
      timestamp: new Date().toISOString(),
    };

    this.log("debug", `Weights uploaded by participant ${participantId + 1}`, uploadResult.metrics);

    return uploadResult;
  }

  // ============================================================
  // GCN-specific training methods (graph datasets like Cora)
  // ============================================================

  async gcnLocalTraining(participantId, trainIdx, globalWeights, participantAddress, startTime) {
    this.log("info", `GCN local training for participant ${participantId + 1}, ${trainIdx.length} train nodes`);

    const { features, adj, labels } = this.graphTensors;
    const numClasses = this.graphData.numClasses;
    const weightDecay = 1e-4;

    const localModel = createGCNModel(this.graphData.numFeatures, numClasses, 16, this.learningRate, 0.5);
    localModel.setWeights(globalWeights);
    localModel.train();

    const optimizer = tf.train.adam(this.learningRate, 0.9, 0.999, 1e-8);
    const trainIdxTensor = tf.tensor1d(trainIdx, "int32");

    let finalLoss = -1;
    let finalAccuracy = -1;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const trainableVars = localModel.getTrainableVariables();
      const lossFunc = () => {
        const output = localModel.forward(features, adj);
        const trainOutput = tf.gather(output, trainIdxTensor);
        const trainLabels = tf.gather(labels, trainIdxTensor);
        const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(trainLabels, numClasses), trainOutput);
        const l2Loss = tf
          .add(tf.sum(tf.square(localModel.gc1.weight)), tf.sum(tf.square(localModel.gc2.weight)))
          .mul(weightDecay / 2);
        return tf.add(loss, l2Loss);
      };
      const lossVal = optimizer.minimize(lossFunc, true, trainableVars);
      if (epoch === this.epochs - 1) {
        finalLoss = lossVal ? lossVal.dataSync()[0] : -1;
      }
      if (lossVal) lossVal.dispose();
    }

    // Evaluate on train nodes
    localModel.eval();
    const evalOutput = localModel.forward(features, adj);
    const trainOutput = tf.gather(evalOutput, trainIdxTensor);
    const trainLabels = tf.gather(labels, trainIdxTensor);
    const predictions = tf.argMax(trainOutput, 1);
    const correct = tf.sum(tf.cast(tf.equal(predictions, trainLabels), "float32"));
    finalAccuracy = correct.dataSync()[0] / trainIdx.length;

    // Dispose eval tensors
    evalOutput.dispose();
    trainOutput.dispose();
    trainLabels.dispose();
    predictions.dispose();
    correct.dispose();
    trainIdxTensor.dispose();

    const gcnWeights = localModel.getWeights();
    localModel.dispose();

    const trainingTime = (Date.now() - startTime) / 1000;
    const trainingMetrics = {
      participantId: participantId + 1,
      loss: finalLoss,
      accuracy: finalAccuracy,
      samples: trainIdx.length,
    };

    this.logCSV(
      "participant",
      participantId + 1,
      participantAddress,
      this.trainingHistory.length + 1,
      finalAccuracy,
      finalLoss,
      trainingTime,
      trainIdx.length,
      "success"
    );

    this.log("debug", `GCN local training done for participant ${participantId + 1}`, trainingMetrics);

    // Return GCN weights as plain object (not flat tensor array)
    return {
      weights: gcnWeights,
      metrics: trainingMetrics,
      isGCN: true,
    };
  }

  async gcnAggregate(participantResults, ownerAddress) {
    this.log("info", `GCN FedAvg aggregation for ${participantResults.length} participants`);
    const startTime = Date.now();

    const numParticipants = participantResults.length;
    const totalSamples = participantResults.reduce((s, p) => s + p.metrics.samples, 0);

    // Weighted average of GCN weights
    const layers = ["gc1", "gc2"];
    const parts = ["weight", "bias"];

    const aggregated = { gc1: { weight: null, bias: null }, gc2: { weight: null, bias: null } };

    for (let i = 0; i < numParticipants; i++) {
      const scale = participantResults[i].metrics.samples / totalSamples;
      const w = participantResults[i].weights;

      for (const layer of layers) {
        for (const part of parts) {
          const contrib = tf.mul(tf.tensor(w[layer][part]), scale);
          if (aggregated[layer][part] === null) {
            aggregated[layer][part] = contrib;
          } else {
            const prev = aggregated[layer][part];
            aggregated[layer][part] = tf.add(prev, contrib);
            prev.dispose();
          }
        }
      }
    }

    // Convert to plain arrays and set on global model
    const finalWeights = {
      gc1: {
        weight: aggregated.gc1.weight.arraySync(),
        bias: aggregated.gc1.bias.arraySync(),
      },
      gc2: {
        weight: aggregated.gc2.weight.arraySync(),
        bias: aggregated.gc2.bias.arraySync(),
      },
    };

    for (const layer of layers) {
      for (const part of parts) {
        aggregated[layer][part].dispose();
      }
    }

    this.model.setWeights(finalWeights);

    const avgMetrics = {
      avgLoss: participantResults.reduce((s, p) => s + p.metrics.loss, 0) / numParticipants,
      avgAccuracy: participantResults.reduce((s, p) => s + p.metrics.accuracy, 0) / numParticipants,
      totalSamples,
    };

    const aggregationTime = (Date.now() - startTime) / 1000;
    const aggregationResult = {
      round: this.trainingHistory.length + 1,
      numParticipants,
      avgMetrics,
      timestamp: new Date().toISOString(),
    };
    this.aggregationHistory.push(aggregationResult);

    this.logCSV(
      "aggregator",
      "aggregate",
      ownerAddress,
      this.trainingHistory.length + 1,
      avgMetrics.avgAccuracy,
      avgMetrics.avgLoss,
      aggregationTime,
      avgMetrics.totalSamples,
      "success"
    );

    this.log("debug", "GCN FedAvg aggregation complete", aggregationResult);
    // Return GCN weights as plain object for publishModel compatibility
    return finalWeights;
  }

  async gcnEvaluateModel(ownerAddress) {
    this.log("info", "Evaluating GCN global model on test nodes");
    const startTime = Date.now();

    const { features, adj, labels } = this.graphTensors;
    const numClasses = this.graphData.numClasses;
    const idxTest = this.graphData.idxTest;
    const idxTestTensor = tf.tensor1d(idxTest, "int32");

    this.model.eval();
    const output = this.model.forward(features, adj);
    const testOutput = tf.gather(output, idxTestTensor);
    const testLabels = tf.gather(labels, idxTestTensor);

    const lossTensor = tf.losses.softmaxCrossEntropy(tf.oneHot(testLabels, numClasses), testOutput);
    const testLoss = lossTensor.dataSync()[0];

    const predictions = tf.argMax(testOutput, 1);
    const correct = tf.sum(tf.cast(tf.equal(predictions, testLabels), "float32"));
    const testAccuracy = correct.dataSync()[0] / idxTest.length;

    output.dispose();
    testOutput.dispose();
    testLabels.dispose();
    lossTensor.dispose();
    predictions.dispose();
    correct.dispose();
    idxTestTensor.dispose();

    const evaluationTime = (Date.now() - startTime) / 1000;
    const evaluationResult = {
      testLoss,
      testAccuracy,
      testSamples: idxTest.length,
      evaluationTime,
      timestamp: new Date().toISOString(),
    };

    this.logCSV(
      "aggregator",
      "evaluate",
      ownerAddress,
      this.trainingHistory.length + 1,
      testAccuracy,
      testLoss,
      evaluationTime,
      idxTest.length,
      "success"
    );

    this.log("info", `GCN evaluation: acc=${(testAccuracy * 100).toFixed(2)}%, loss=${testLoss.toFixed(4)}`);
    return evaluationResult;
  }

  async aggregate(participantResults, ownerAddress) {
    this.log("info", "Aggregating weights from all participants");

    if (participantResults.length === 0) {
      throw new Error("No participant results to aggregate");
    }

    if (this.modelName === "gcn") {
      return await this.gcnAggregate(participantResults, ownerAddress);
    }

    const numParticipants = participantResults.length;
    const startTime = Date.now();

    try {
      this.log("info", `Performing FedAvg aggregation for ${numParticipants} participants`);

      // Get the structure of weights from the first participant
      const firstWeights = participantResults[0].weights;
      const aggregatedWeights = [];

      // Perform Federated Averaging (FedAvg) for each layer
      for (let layerIdx = 0; layerIdx < firstWeights.length; layerIdx++) {
        this.log("debug", `Aggregating layer ${layerIdx + 1}/${firstWeights.length}`);

        // Extract tensor data and perform manual aggregation
        const layerShape = firstWeights[layerIdx].shape;
        const aggregatedData = new Float32Array(firstWeights[layerIdx].data.length);

        // Sum all participant weights for this layer
        for (let participantIdx = 0; participantIdx < numParticipants; participantIdx++) {
          const participantWeights = participantResults[participantIdx].weights;
          const participantLayer = participantWeights[layerIdx];
          const participantData = participantLayer.data;

          // Add participant's layer weights to the aggregated sum
          for (let i = 0; i < aggregatedData.length; i++) {
            aggregatedData[i] += participantData[i];
          }
        }

        // Divide by number of participants to get average
        for (let i = 0; i < aggregatedData.length; i++) {
          aggregatedData[i] /= numParticipants;
        }

        // Create new tensor from aggregated data
        const averageLayer = tf.tensor(aggregatedData, layerShape);
        aggregatedWeights.push(averageLayer);
      }

      // Update global model with aggregated weights
      this.model.setWeights(aggregatedWeights);
      this.log("info", "Global model updated with aggregated weights");

      // Calculate average metrics
      const avgMetrics = {
        avgLoss: participantResults.reduce((sum, p) => sum + p.metrics.loss, 0) / numParticipants,
        avgAccuracy: participantResults.reduce((sum, p) => sum + p.metrics.accuracy, 0) / numParticipants,
        totalSamples: participantResults.reduce((sum, p) => sum + p.metrics.samples, 0),
      };

      const aggregationTime = (Date.now() - startTime) / 1000; // Convert to seconds

      const aggregationResult = {
        round: this.trainingHistory.length + 1,
        numParticipants,
        avgMetrics,
        timestamp: new Date().toISOString(),
      };

      this.aggregationHistory.push(aggregationResult);

      // Log CSV data for aggregation
      this.logCSV(
        "aggregator",
        "aggregate",
        ownerAddress,
        this.trainingHistory.length + 1, // Current round
        avgMetrics.avgAccuracy,
        avgMetrics.avgLoss,
        aggregationTime,
        avgMetrics.totalSamples,
        "success"
      );

      this.log("debug", "FedAvg aggregation completed successfully", aggregationResult);

      return aggregatedWeights;
    } catch (error) {
      const aggregationTime = (Date.now() - startTime) / 1000;

      // Log CSV data for failed aggregation
      this.logCSV(
        "aggregator",
        "aggregate",
        ownerAddress,
        this.trainingHistory.length + 1,
        -1,
        -1,
        aggregationTime,
        0,
        "error",
        error.message
      );

      this.log("error", "Error during aggregation", error);
      throw error;
    }
  }

  async evaluateModel(ownerAddress) {
    this.log("info", "Evaluating global model on test data");

    if (this.modelName === "gcn") {
      return await this.gcnEvaluateModel(ownerAddress);
    }

    const startTime = Date.now();

    try {
      const processedTestData = this.preprocessData(this.testData);

      const evaluation = await this.model.evaluate(processedTestData.images, processedTestData.labels, {
        batchSize: this.batchSize,
        verbose: 0,
      });

      const testLoss = evaluation[0].dataSync()[0];
      const testAccuracy = evaluation[1].dataSync()[0];

      // Clean up tensors
      processedTestData.images.dispose();
      processedTestData.labels.dispose();
      evaluation[0].dispose();
      evaluation[1].dispose();

      const evaluationTime = (Date.now() - startTime) / 1000; // Convert to seconds

      const evaluationResult = {
        testLoss,
        testAccuracy,
        testSamples: this.testData.length,
        evaluationTime,
        timestamp: new Date().toISOString(),
      };

      // Log CSV data for model evaluation
      this.logCSV(
        "aggregator",
        "evaluate",
        ownerAddress,
        this.trainingHistory.length + 1, // Current round
        testAccuracy,
        testLoss,
        evaluationTime,
        this.testData.length,
        "success"
      );

      this.log("debug", `Model evaluation completed in ${evaluationTime.toFixed(2)}s`, evaluationResult);

      return evaluationResult;
    } catch (error) {
      const evaluationTime = (Date.now() - startTime) / 1000;

      // Log CSV data for failed evaluation
      this.logCSV(
        "aggregator",
        "evaluate",
        ownerAddress,
        this.trainingHistory.length + 1,
        -1,
        -1,
        evaluationTime,
        this.testData.length,
        "error",
        error.message
      );

      this.log("error", "Error during model evaluation", error);
      throw error;
    }
  }

  async runParallelLocalTraining(globalWeights, participantAddresses) {
    const startTime = Date.now();
    this.log("info", `Starting parallel local training for ${this.participantCount} participants`);

    // Create training tasks for all participants
    const trainingTasks = [];
    for (let participantId = 0; participantId < this.participantCount; participantId++) {
      this.log("debug", `Queuing training task for participant ${participantId + 1}`);
      const participantAddress = participantAddresses[participantId];
      const task = this.localTraining(participantId, this.participantData[participantId], globalWeights, participantAddress);
      trainingTasks.push({ participantId, task });
    }

    // Execute all training tasks in parallel with progress tracking
    this.log("info", `Executing ${this.participantCount} training tasks in parallel...`);

    // Use Promise.allSettled to handle any potential failures gracefully
    const trainingResults = await Promise.allSettled(trainingTasks.map((t) => t.task));

    // Check for any failed training tasks
    const failedTasks = trainingResults.filter((result, index) => result.status === "rejected");
    if (failedTasks.length > 0) {
      this.log("error", `${failedTasks.length} training tasks failed`);
      failedTasks.forEach((result, index) => {
        this.log("error", `Participant ${trainingTasks[index].participantId + 1} training failed:`, result.reason);
      });
      throw new Error(`${failedTasks.length} training tasks failed`);
    }

    // Extract successful results
    const successfulResults = trainingResults.map((result) => result.value);

    // Process results and create upload results
    const participantResults = [];
    for (let participantId = 0; participantId < this.participantCount; participantId++) {
      const uploadResult = await this.uploadWeights(participantId, successfulResults[participantId]);
      participantResults.push(uploadResult);
    }

    const totalTime = (Date.now() - startTime) / 1000;
    this.log("info", `Parallel local training completed for ${this.participantCount} participants in ${totalTime.toFixed(2)}s`);

    return participantResults;
  }

  async runSequentialLocalTraining(globalWeights, participantAddresses) {
    const startTime = Date.now();
    this.log("info", `Starting sequential local training for ${this.participantCount} participants`);

    const participantResults = [];
    for (let participantId = 0; participantId < this.participantCount; participantId++) {
      const participantStartTime = Date.now();
      this.log("info", `--- Participant ${participantId + 1} Training ---`);

      const participantAddress = participantAddresses[participantId];
      const localTrainingResult = await this.localTraining(
        participantId,
        this.participantData[participantId],
        globalWeights,
        participantAddress
      );
      const uploadResult = await this.uploadWeights(participantId, localTrainingResult);
      participantResults.push(uploadResult);

      const participantTime = (Date.now() - participantStartTime) / 1000;
      this.log("info", `--- Participant ${participantId + 1} completed in ${participantTime.toFixed(2)}s ---`);
    }

    const totalTime = (Date.now() - startTime) / 1000;
    this.log("info", `Sequential local training completed for ${this.participantCount} participants in ${totalTime.toFixed(2)}s`);

    return participantResults;
  }

  getTrainingSummary() {
    const summary = {
      totalRounds: this.rounds,
      participantCount: this.participantCount,
      finalTestAccuracy: this.trainingHistory[this.trainingHistory.length - 1]?.evaluationResult.testAccuracy || 0,
      finalTestLoss: this.trainingHistory[this.trainingHistory.length - 1]?.evaluationResult.testLoss || 0,
      trainingHistory: this.trainingHistory,
      aggregationHistory: this.aggregationHistory,
    };

    return summary;
  }

  async cleanup() {
    this.log("info", "Cleaning up federated learning trainer");

    if (this.model) {
      this.model.dispose();
    }

    // Clean up any remaining tensors
    tf.tidy(() => {});

    this.log("info", "Cleanup completed");
  }
}

module.exports = { FederatedLearningTrainer };

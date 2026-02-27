const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const os = require("os");
class FederatedLearningTrainerPy {
  constructor(options = {}) {
    this.dataDir = options.dataDir || path.join(__dirname, "..", "..", "target", "data");
    this.batchSize = options.localBatchSize || options.batchSize || 32;
    this.epochs = options.epochs || 1;
    this.learningRate = options.learningRate || 0.001;
    this.logLevel = options.logLevel || "info";
    this.useParallelTraining = options.useParallelTraining !== false;
    this.dataset = options.dataset;
    this.modelName = options.model;
    this.participantCount = options.participantCount;
    this.maxDataSizePerParticipant = options.maxDataSizePerParticipant || 1000;
    this.maxParallelWorkersPerGpu = options.maxParallelWorkersPerGpu || 1;
    this.maxRetries = options.maxRetries || 3; // Default to 3 retries if not provided
    this.trainingHistory = [];

    // CSV logging system
    this.csvLog = [];
    this.runId = this.generateRunId();
    this.startTime = new Date();

    this.workerPath = path.join(__dirname, "worker", "worker.py");
    this.tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "fl-weights-"));
  }

  generateRunId() {
    return `fl_run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  log(level, message, data = null) {
    if (this.logLevel === "none") return;
    if (this.logLevel !== "debug" && level === "debug") return;
    console.log(`[${level.toUpperCase()}] ${message}`);
    if (data) console.log(JSON.stringify(data, null, 2));
  }

  async initialize() {
    this.log("info", "Python-backed Federated Learning Trainer initialized.");
    // In a real scenario, you might check for python/dependency existence here
    return Promise.resolve();
  }

  runWorker(args) {
    this.log("debug", `Spawning Python worker: python3 ${this.workerPath} ${args.join(" ")}`);
    return new Promise((resolve, reject) => {
      const python = spawn("python3", [this.workerPath, ...args]);

      python.stdout.on("data", (data) => {
        process.stdout.write(data.toString());
      });

      python.stderr.on("data", (data) => {
        process.stderr.write(data.toString());
      });

      python.on("close", (code) => {
        if (code !== 0) {
          this.log("error", `Python worker script exited with code ${code}`);
          return reject(new Error(`Python worker failed with code ${code}`));
        }
        resolve();
      });
    });
  }

  async publishModel() {
    // The Python worker manages its own model, but we need a dummy file path for the first round.
    const initialWeightsPath = path.join(this.tempDir, "initial_weights.pth");
    // Create an empty file, as the first training round doesn't need input weights.
    fs.writeFileSync(initialWeightsPath, "");
    return initialWeightsPath;
  }

  async uploadWeights(participantId, localTrainingResult) {
    // In this backend, the "upload" is just passing the file path.
    return {
      participantId: participantId + 1,
      weights: localTrainingResult.weights,
      metrics: localTrainingResult.metrics,
    };
  }

  async aggregate(participantResults, ownerAddress) {
    this.log("info", "[JS] Aggregating weights using Python worker");
    const startTime = Date.now();
    const participantWeightFiles = participantResults.map((r) => r.weights);
    const aggregatedWeightsPath = path.join(this.tempDir, `aggregated_round_${Date.now()}.pth`);

    this.log("debug", `  - Aggregating ${participantWeightFiles.length} weight files.`);
    this.log("debug", `  - Output aggregated weights: ${aggregatedWeightsPath}`);
    const args = [
      "aggregate",
      "--model",
      this.modelName, // Model is needed to know the structure, though not used in aggregation logic itself
      "--participant-weights-in",
      ...participantWeightFiles,
      "--weights-out",
      aggregatedWeightsPath,
    ];

    try {
      await this.runWorker(args);

      const aggregationTime = (Date.now() - startTime) / 1000;

      // Calculate average metrics from participant results
      const numParticipants = participantResults.length;
      const avgAccuracy = participantResults.reduce((sum, p) => sum + (p.metrics.accuracy || 0), 0) / numParticipants;
      const avgLoss = participantResults.reduce((sum, p) => sum + (p.metrics.loss || 0), 0) / numParticipants;
      const totalSamples = participantResults.reduce((sum, p) => sum + (p.metrics.samples || 0), 0);

      // Log CSV data for aggregation
      this.logCSV(
        "aggregator",
        "aggregate",
        ownerAddress,
        this.trainingHistory.length + 1,
        avgAccuracy,
        avgLoss,
        aggregationTime,
        totalSamples,
        "success"
      );

      return aggregatedWeightsPath; // Return path to the new global weights
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

      throw error;
    }
  }

  async evaluateModel(ownerAddress, globalWeightsPath) {
    this.log("info", "[JS] Evaluating global model using Python worker");
    const startTime = Date.now();
    const resultsOutPath = path.join(this.tempDir, `eval_results_${Date.now()}.json`);

    this.log("debug", `  - Evaluating with weights: ${globalWeightsPath}`);
    this.log("debug", `  - Results will be saved to: ${resultsOutPath}`);

    const args = [
      "evaluate",
      "--model",
      this.modelName,
      "--dataset",
      this.dataset,
      "--weights-in",
      globalWeightsPath,
      "--results-out",
      resultsOutPath,
    ];

    try {
      await this.runWorker(args);
      const evaluationResult = JSON.parse(fs.readFileSync(resultsOutPath, "utf-8"));

      const evaluationTime = (Date.now() - startTime) / 1000;

      // Log CSV data for model evaluation
      this.logCSV(
        "aggregator",
        "evaluate",
        ownerAddress,
        this.trainingHistory.length + 1,
        evaluationResult.testAccuracy || evaluationResult.accuracy || -1,
        evaluationResult.testLoss || evaluationResult.loss || -1,
        evaluationTime,
        evaluationResult.testSamples || evaluationResult.samples || 0,
        "success"
      );

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
        0,
        "error",
        error.message
      );

      throw error;
    }
  }

  // --- Training Execution ---
  _getGpuIds() {
    const gpuIdsPath = path.join(__dirname, "..", "..", ".gpuids");
    try {
      if (fs.existsSync(gpuIdsPath)) {
        const content = fs.readFileSync(gpuIdsPath, "utf8").trim();
        if (content) {
          const ids = content.split(",").map((id) => parseInt(id.trim(), 10));
          if (ids.length > 0) return ids;
        }
      }
    } catch (error) {
      this.log("warn", `Could not read or parse .gpuids file, defaulting to [0]. Error: ${error.message}`);
    }
    return [0]; // Default to GPU 0
  }

  async runParallelLocalTraining(globalWeightsPath, participantAddresses) {
    this.log("info", `[JS] Starting PARALLEL local training for ${participantAddresses.length} participants.`);
    const allResults = [];
    const participantIds = Array.from({ length: participantAddresses.length }, (_, k) => k);

    let i = 0;
    while (i < participantIds.length) {
      const gpuIds = this._getGpuIds();
      const maxParallelWorkers = gpuIds.length * this.maxParallelWorkersPerGpu;
      this.log("info", `  - Discovered ${gpuIds.length} GPUs. Max concurrent workers: ${maxParallelWorkers}`);

      let batchIds = participantIds.slice(i, i + maxParallelWorkers);
      this.log("info", `  - Processing batch of ${batchIds.length} participants (from index ${i})`);

      let retryCount = 0;
      while (batchIds.length > 0 && retryCount <= this.maxRetries) {
        if (retryCount > 0) {
          this.log("warn", `  - Retrying ${batchIds.length} failed tasks (Attempt ${retryCount}/${this.maxRetries})`);
        }

        const tasks = batchIds.map((id) =>
          this._runSingleTrainingInstance(id, globalWeightsPath, gpuIds, participantAddresses[id])
        );
        const results = await Promise.all(tasks);

        const successfulResults = results.filter((r) => r.status === "success").map((r) => r.result);
        allResults.push(...successfulResults);

        batchIds = results.filter((r) => r.status === "error").map((r) => r.participantId);

        if (batchIds.length > 0) {
          retryCount++;
          if (retryCount > this.maxRetries) {
            throw new Error(`Exceeded max retries for participants: ${batchIds.join(", ")}. Terminating.`);
          }
          await new Promise((resolve) => setTimeout(resolve, 1000 * retryCount));
        }
      }
      i += maxParallelWorkers;
    }
    return allResults;
  }

  async runSequentialLocalTraining(globalWeightsPath, participantAddresses) {
    this.log("info", `[JS] Starting SEQUENTIAL local training for ${participantAddresses.length} participants.`);
    const participantResults = [];
    const gpuIds = this._getGpuIds();
    for (let i = 0; i < participantAddresses.length; i++) {
      const result = await this._runSingleTrainingInstance(i, globalWeightsPath, gpuIds, participantAddresses[i]);
      if (result.status === "success") {
        participantResults.push(result.result);
      } else {
        // In sequential mode, we could also implement retries, but for now we'll just throw.
        throw new Error(`Sequential training failed for participant ${i + 1}`);
      }
    }
    return participantResults;
  }

  async _runSingleTrainingInstance(participantId, globalWeightsPath, gpuIds, participantAddress = null) {
    const gpuId = gpuIds[participantId % gpuIds.length];
    this.log("debug", `  - Spawning worker for participant ${participantId + 1} on GPU ${gpuId}`);

    const startTime = Date.now();
    const weightsOutPath = path.join(this.tempDir, `participant_${participantId + 1}_weights.pth`);
    const resultsOutPath = path.join(this.tempDir, `participant_${participantId + 1}_results.json`);

    const args = [
      "train",
      "--model",
      this.modelName,
      "--dataset",
      this.dataset,
      "--epochs",
      this.epochs,
      "--learning-rate",
      this.learningRate,
      "--participant-id",
      participantId,
      "--participant-count",
      this.participantCount,
      "--max-data-size",
      this.maxDataSizePerParticipant,
      "--gpu-id",
      gpuId,
      "--weights-in",
      globalWeightsPath,
      "--weights-out",
      weightsOutPath,
      "--results-out",
      resultsOutPath,
    ];

    try {
      await this.runWorker(args);
      const metrics = JSON.parse(fs.readFileSync(resultsOutPath, "utf-8"));

      const trainingTime = (Date.now() - startTime) / 1000;

      // Log CSV data for participant training
      this.logCSV(
        "participant",
        participantId + 1,
        participantAddress || `participant_${participantId + 1}`,
        this.trainingHistory.length + 1,
        metrics.accuracy || -1,
        metrics.loss || -1,
        trainingTime,
        metrics.samples || 0,
        "success"
      );

      return {
        status: "success",
        result: {
          weights: weightsOutPath,
          metrics: {
            ...metrics,
            participantId: participantId + 1,
            samples: metrics.samples || 0,
          },
        },
      };
    } catch (error) {
      const trainingTime = (Date.now() - startTime) / 1000;

      // Log CSV data for failed training
      this.logCSV(
        "participant",
        participantId + 1,
        participantAddress || `participant_${participantId + 1}`,
        this.trainingHistory.length + 1,
        -1,
        -1,
        trainingTime,
        0,
        "error",
        error.message
      );

      this.log("error", `  - Worker for participant ${participantId + 1} failed: ${error.message}`);
      return { status: "error", participantId };
    }
  }

  addRoundToHistory(roundResult) {
    this.trainingHistory.push(roundResult);
  }

  getTrainingSummary() {
    if (this.trainingHistory.length === 0) {
      return {
        totalRounds: 0,
        participantCount: this.participantCount,
        finalTestAccuracy: 0,
        finalTestLoss: 0,
      };
    }

    const lastRound = this.trainingHistory[this.trainingHistory.length - 1];
    return {
      totalRounds: this.trainingHistory.length,
      participantCount: this.participantCount,
      finalTestAccuracy: lastRound.evaluationResult.testAccuracy,
      finalTestLoss: lastRound.evaluationResult.testLoss,
    };
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
      participant_count: this.participantCount,
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

  cleanup() {
    fs.rmSync(this.tempDir, { recursive: true, force: true });
    this.log("info", "Cleaned up temporary Python worker files.");
  }
}

module.exports = { FederatedLearningTrainerPy };

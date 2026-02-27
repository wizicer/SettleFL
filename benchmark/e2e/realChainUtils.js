// Load environment variables from .env file
require("dotenv").config();

const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");

// Import the original utilities but we'll need to adapt them
const { TestUtils } = require("../../test/utils/testUtils.js");
const { RewardMatrixHelper } = require("../../test/utils/RewardMatrixHelper.js");
const { ChallengeProver } = require("../../test/prover/challengeProver.js");
const { ChallengeProverRs } = require("../../test/prover/challengeProverRs.js");
const { CounterProver } = require("../../test/prover/counterProver.js");
const { CounterProverRs } = require("../../test/prover/counterProverRs.js");
const { RedeemOneShotProver } = require("../../test/prover/redeemOneShotProver.js");
const { RedeemOneShotProverRs } = require("../../test/prover/redeemOneShotProverRs.js");
const { RedeemMultiShotProver } = require("../../test/prover/redeemMultiShotProver.js");
const { RedeemMultiShotProverRs } = require("../../test/prover/redeemMultiShotProverRs.js");
const { CommitProver } = require("../../test/prover/commitProver.js");
const { CommitProverRs } = require("../../test/prover/commitProverRs.js");
const { CircuitCompiler } = require("../../test/utils/circuitCompiler.js");
// Dynamically import the trainer based on config
const { FederatedLearningTrainer } = require("./federatedLearningTrainer.js");
const { FederatedLearningTrainerPy } = require("./federatedLearningTrainer.py.js");

const JobStatus = {
  0: "Committed",
  1: "RewardInit",
  2: "Challenged",
  3: "Paying",
  4: "Paid",
};

function generateWallets(count) {
  const wallets = [];
  for (let i = 0; i < count; i++) {
    const wallet = ethers.Wallet.createRandom();
    wallets.push(wallet);
  }
  return wallets;
}

class RealChainUtils {
  constructor(options = {}) {
    this.testUtils = null;
    this.circuitCompiler = new CircuitCompiler();
    this.proverBackend = options.proverBackend || "rapidsnark"; // Default to Rapidsnark

    // Allow custom circuit paths to be passed in
    this.challengeWasmPath = options.challengeWasmPath || path.join(__dirname, "..", "circuits_generator", "Challenge.wasm");
    this.challengeZkeyPath =
      options.challengeZkeyPath || path.join(__dirname, "..", "circuits_generator", "Challenge_final.zkey");
    this.counterWasmPath = options.counterWasmPath || path.join(__dirname, "..", "circuits_generator", "Counter.wasm");
    this.counterZkeyPath = options.counterZkeyPath || path.join(__dirname, "..", "circuits_generator", "Counter_final.zkey");
    this.redeemOneShotWasmPath =
      options.redeemOneShotWasmPath || path.join(__dirname, "..", "circuits_generator", "RedeemOneShot.wasm");
    this.redeemOneShotZkeyPath =
      options.redeemOneShotZkeyPath || path.join(__dirname, "..", "circuits_generator", "RedeemOneShot_final.zkey");
    this.redeemMultiShotWasmPath =
      options.redeemMultiShotWasmPath || path.join(__dirname, "..", "circuits_generator", "RedeemMultiShot.wasm");
    this.redeemMultiShotZkeyPath =
      options.redeemMultiShotZkeyPath || path.join(__dirname, "..", "circuits_generator", "RedeemMultiShot_final.zkey");
    this.commitWasmPath = options.commitWasmPath || path.join(__dirname, "..", "..", "test", "circuits_generator", "Commit.wasm");
    this.commitZkeyPath =
      options.commitZkeyPath || path.join(__dirname, "..", "..", "test", "circuits_generator", "Commit_final.zkey");
  }

  // Helper function to create the appropriate prover based on backend
  createProver(proverType, maxRoundCount, maxParticipantCount, chunkSize, payoutBatchSize, wasmPath, zkeyPath) {
    console.log(`    Using ${this.proverBackend} backend for ${proverType} proving`);

    if (this.proverBackend === "rapidsnark") {
      // Check if Rapidsnark is available
      if (!process.env.RAPID_SNARK_BIN_PATH) {
        console.log(`    ⚠️ Rapidsnark not available (RAPID_SNARK_BIN_PATH not set), falling back to SnarkJS`);
        return this.createSnarkJSProver(
          proverType,
          maxRoundCount,
          maxParticipantCount,
          chunkSize,
          payoutBatchSize,
          wasmPath,
          zkeyPath
        );
      }
      return this.createRapidsnarkProver(
        proverType,
        maxRoundCount,
        maxParticipantCount,
        chunkSize,
        payoutBatchSize,
        wasmPath,
        zkeyPath
      );
    } else if (this.proverBackend === "snarkjs") {
      return this.createSnarkJSProver(
        proverType,
        maxRoundCount,
        maxParticipantCount,
        chunkSize,
        payoutBatchSize,
        wasmPath,
        zkeyPath
      );
    } else {
      throw new Error(`Unknown prover backend: ${this.proverBackend}. Available: rapidsnark, snarkjs`);
    }
  }

  createSnarkJSProver(proverType, maxRoundCount, maxParticipantCount, chunkSize, payoutBatchSize, wasmPath, zkeyPath) {
    switch (proverType) {
      case "challenge":
        return new ChallengeProver(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      case "counter":
        return new CounterProver(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      case "redeemOneShot":
        return new RedeemOneShotProver(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      case "redeemMultiShot":
        return new RedeemMultiShotProver(maxParticipantCount, maxRoundCount, chunkSize, payoutBatchSize, wasmPath, zkeyPath);
      case "commit":
        return new CommitProver(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      default:
        throw new Error(`Unknown prover type: ${proverType}`);
    }
  }

  createRapidsnarkProver(proverType, maxRoundCount, maxParticipantCount, chunkSize, payoutBatchSize, wasmPath, zkeyPath) {
    switch (proverType) {
      case "challenge":
        return new ChallengeProverRs(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      case "counter":
        return new CounterProverRs(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      case "redeemOneShot":
        return new RedeemOneShotProverRs(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      case "redeemMultiShot":
        return new RedeemMultiShotProverRs(maxParticipantCount, maxRoundCount, chunkSize, payoutBatchSize, wasmPath, zkeyPath);
      case "commit":
        return new CommitProverRs(maxRoundCount, maxParticipantCount, chunkSize, wasmPath, zkeyPath);
      default:
        throw new Error(`Unknown prover type: ${proverType}`);
    }
  }

  async initialize() {
    if (!this.testUtils) {
      this.testUtils = new TestUtils();
      await this.testUtils.initialize();
    }
  }

  async setupJob(config, flzkpReward, owner, participantAddresses, roundCommitCallback, extraCommitCallback) {
    console.log("\n🤖 Initializing Federated Learning Training System");

    // Initialize federated learning trainer
    const TrainerClass = config.usePythonTrainer ? FederatedLearningTrainerPy : FederatedLearningTrainer;
    this.flTrainer = new TrainerClass({
      participantCount: participantAddresses.length,
      rounds: config.commitRounds,
      batchSize: config.localBatchSize,
      epochs: config.epochs,
      learningRate: config.learningRate,
      logLevel: config.logLevel,
      useParallelTraining: config.useParallelTraining !== false, // Use config setting, default to true
      dataset: config.dataset, // Pass dataset option
      model: config.model, // Pass model option
      maxDataSizePerParticipant: config.maxDataSizePerParticipant,
      maxParallelWorkersPerGpu: config.maxParallelWorkersPerGpu,
      maxRetries: config.maxRetries, // Pass retry limit
    });

    await this.flTrainer.initialize();

    const helper = new RewardMatrixHelper(config.maxRoundCount, config.maxParticipantCount, config.chunkSize);
    await helper.init();

    const publicKey = helper.getPubkey(config.privateKey);
    const pubHash = helper.hashPubkey(publicKey);

    console.log("📋 Creating new federated learning job on blockchain");
    const tx = await flzkpReward.connect(owner).new_job(pubHash, BigInt(config.salt), {
      value: config.totalReward,
      gasPrice: config.gasPrice,
    });
    const receipt = await tx.wait();
    const gasUsed = receipt.gasUsed;

    const jobId = 1;
    let matrix = helper.emptyMatrix();

    const commitments = [];
    const commitGasCosts = [];
    let currentRound = 1;
    const trainingHistory = [];

    // Detect whether the contract is in commit mode (requires ZK proof per commit)
    const isCommitMode = await flzkpReward.commitMode();
    if (isCommitMode) {
      console.log("🔐 Contract is in commit mode — ZK proofs will be generated for each commit");
    }

    console.log(`🔄 Starting federated learning rounds (${config.commitRounds} rounds)`);

    const overallStartTime = Date.now();

    // Initialize the path for the global weights. This will be updated each round.
    let globalWeightsPath = await this.flTrainer.publishModel();

    for (let i = 0; i < config.commitRounds; i++) {
      const roundStartTime = Date.now();
      console.log(`\n📊 === Federated Learning Round ${i + 1}/${config.commitRounds} ===`);

      // The globalWeightsPath will be updated each round. For the first round, it's the initial (empty) model.
      const trainingStartTime = Date.now();
      console.log(`[realChainUtils] useParallelTraining flag is: ${config.useParallelTraining}`);
      const trainingMethod = config.useParallelTraining ? "runParallelLocalTraining" : "runSequentialLocalTraining";
      console.log(`[realChainUtils] Selected training method: ${trainingMethod}`);

      const participantResults = await this.flTrainer[trainingMethod](globalWeightsPath, participantAddresses);

      // 3. Aggregate weights (Owner)
      console.log("\n🔗 Owner aggregating weights from all participants");
      // The output of aggregation becomes the input for the next round.
      globalWeightsPath = await this.flTrainer.aggregate(participantResults, owner.address);

      // 4. Evaluate global model
      console.log("📈 Evaluating global model performance");
      let evaluationResult;
      try {
        const evaluationStartTime = Date.now();
        evaluationResult = await this.flTrainer.evaluateModel(owner.address, globalWeightsPath);
        const evaluationTime = (Date.now() - evaluationStartTime) / 1000;
        console.log(`⏱️  Model evaluation completed in ${evaluationTime.toFixed(2)}s`);
      } catch (error) {
        console.log("⚠️  Model evaluation failed, using default metrics:", error.message);
        evaluationResult = { testAccuracy: -1, testLoss: -1 };
      }

      // Store round results
      const roundResult = {
        round: i + 1,
        participantResults,
        evaluationResult,
        timestamp: new Date().toISOString(),
      };
      trainingHistory.push(roundResult);

      // Also update the federated learning trainer's training history for CSV logging
      this.flTrainer.addRoundToHistory(roundResult);

      const roundTime = (Date.now() - roundStartTime) / 1000;
      console.log(
        `✅ Round ${i + 1} completed in ${roundTime.toFixed(2)}s - Test Accuracy: ${(evaluationResult.testAccuracy * 100).toFixed(
          2
        )}%`
      );

      // 5. Generate reward matrix row based on training performance
      console.log("💰 Generating reward matrix based on training performance");
      const rewardRow = participantAddresses.map((_, participantId) => {
        const participantResult = participantResults[participantId];
        // Base reward on training accuracy and loss
        const accuracyScore = Math.floor(participantResult.metrics.accuracy * 100);
        const lossPenalty = Math.floor(participantResult.metrics.loss * 10);
        const baseReward = Math.max(1, accuracyScore - lossPenalty);
        return BigInt(baseReward);
      });

      matrix = helper.appendRow(matrix, rewardRow);
      const C = helper.getCommitment(matrix, participantAddresses, BigInt(config.salt));
      commitments.push(helper.uint8ArrayToBigInt(C));

      console.log("📝 Committing round results to blockchain");
      // In commit mode the contract requires a ZK proof alongside the commitment
      let commitTx;
      if (isCommitMode) {
        // C1 is the commitment of the state before this round.
        // For the first round (i=0), P1 is empty (no previous participants).
        // For subsequent rounds, P1 uses all participants.
        const prevMatrix = matrix.slice(0, matrix.length - 1);
        const prevParticipants = i === 0 ? [] : participantAddresses;
        const p1Number = prevParticipants.length;
        const C1_arr = helper.getCommitment(prevMatrix, prevParticipants, BigInt(config.salt));
        const C2_arr = helper.getCommitment(matrix, participantAddresses, BigInt(config.salt));
        const sig_C1 = helper.sign(config.privateKey, C1_arr);
        const sig_C2 = helper.sign(config.privateKey, C2_arr);
        const proofStartTime = Date.now();
        const commitProver = this.createProver(
          "commit",
          config.maxRoundCount,
          config.maxParticipantCount,
          config.chunkSize,
          config.payoutBatchSize,
          this.commitWasmPath,
          this.commitZkeyPath
        );
        // round_number is 0-indexed (i), P1_number is the number of previous participants
        const commitProof = await commitProver.getProof(
          i,
          matrix,
          [sig_C1.R8x, sig_C1.R8y, sig_C1.S],
          [sig_C2.R8x, sig_C2.R8y, sig_C2.S],
          [publicKey.Ax, publicKey.Ay],
          BigInt(config.salt),
          p1Number,
          participantAddresses.map((addr) => BigInt(addr))
        );
        const proofStruct = { a: commitProof.a, b: commitProof.b, c: commitProof.c };
        const proofEndTime = Date.now();
        const commitStartTime = Date.now();
        commitTx = await flzkpReward
          .connect(owner)
          ["commit(uint256,uint256,(uint256[2],uint256[2][2],uint256[2]))"](jobId, commitments[i], proofStruct, {
            gasPrice: config.gasPrice,
          });
        const commitEndTime = Date.now();
        console.log(`  Proof time: ${proofEndTime - proofStartTime}ms`);
        console.log(`  Commit time: ${commitEndTime - commitStartTime}ms`);
      } else {
        const commitStartTime = Date.now();
        commitTx = await flzkpReward
          .connect(owner)
          ["commit(uint256,uint256)"](jobId, commitments[i], { gasPrice: config.gasPrice });
        const commitEndTime = Date.now();
        console.log(`  Commit time: ${commitEndTime - commitStartTime}ms`);
      }
      const commitReceipt = await commitTx.wait();
      commitGasCosts.push(Number(commitReceipt.gasUsed));

      if (roundCommitCallback) {
        await roundCommitCallback(i, jobId, matrix, helper);
      }
      currentRound++;
    }

    if (extraCommitCallback) {
      const commitReceipt = await extraCommitCallback(jobId, matrix, helper);
      commitGasCosts.push(Number(commitReceipt.gasUsed));
      currentRound++;
    }

    const rewards = helper.calculateRewards(matrix);
    console.log("\n🎯 Initializing reward distribution");
    const rewardInitTx = await flzkpReward.connect(owner).reward_init(jobId, { gasPrice: config.gasPrice });
    const rewardInitReceipt = await rewardInitTx.wait();
    const rewardInitGas = Number(rewardInitReceipt.gasUsed);

    const overallTime = (Date.now() - overallStartTime) / 1000;

    // Get final training summary
    const trainingSummary = this.flTrainer.getTrainingSummary();
    console.log("\n📊 === Federated Learning Training Summary ===");
    console.log(`Total Rounds: ${trainingSummary.totalRounds}`);
    console.log(`Participants: ${trainingSummary.participantCount}`);
    console.log(`Final Test Accuracy: ${(trainingSummary.finalTestAccuracy * 100).toFixed(2)}%`);
    console.log(`Final Test Loss: ${trainingSummary.finalTestLoss.toFixed(4)}`);
    console.log(`⏱️  Total Training Time: ${overallTime.toFixed(2)}s`);
    console.log(`⏱️  Average Time per Round: ${(overallTime / config.commitRounds).toFixed(2)}s`);

    // Cleanup
    await this.flTrainer.cleanup();

    return {
      helper,
      publicKey,
      matrix,
      currentRound,
      rewards,
      jobId,
      commitGasCosts,
      rewardInitGas,
      newJobGas: Number(gasUsed),
      commitments,
      trainingHistory,
      trainingSummary,
      timing: {
        totalTime: overallTime,
        averageTimePerRound: overallTime / config.commitRounds,
        rounds: config.commitRounds,
        participants: participantAddresses.length,
      },
    };
  }

  async executeChallenge(
    config,
    flzkpReward,
    challenger,
    helper,
    publicKey,
    matrix,
    currentRound,
    participantAddresses,
    rewards
  ) {
    const challengeRound = currentRound;
    const V = matrix;

    const C = helper.getCommitment(V, participantAddresses, BigInt(config.salt));
    const sig_C = helper.sign(config.privateKey, C);

    const challengeProver = this.createProver(
      "challenge",
      config.maxRoundCount,
      config.maxParticipantCount,
      config.chunkSize,
      config.payoutBatchSize,
      this.challengeWasmPath,
      this.challengeZkeyPath
    );

    // Measure proving time
    const provingStartTime = Date.now();
    const challengeProof = await challengeProver.getProof(
      participantAddresses.map((addr) => BigInt(addr).toString()),
      challengeRound - 1,
      V.map((row) => row.map((val) => val.toString())),
      [sig_C.R8x.toString(), sig_C.R8y.toString(), sig_C.S.toString()],
      [publicKey.Ax.toString(), publicKey.Ay.toString()],
      config.salt.toString()
    );
    const provingTime = Date.now() - provingStartTime;

    const challengeProofStruct = {
      a: challengeProof.a,
      b: challengeProof.b,
      c: challengeProof.c,
    };

    const challengeTx = await flzkpReward
      .connect(challenger)
      .challenge(1, challengeRound, challengeProofStruct, { gasPrice: config.gasPrice });
    await challengeTx.wait();

    return {
      challengeRound,
      challengeData: { V, sig_C },
      proof: challengeProof,
      provingTime,
    };
  }

  async executeCounter(config, flzkpReward, owner, helper, publicKey, matrix, challengeRound, participantAddresses) {
    const counterRound = challengeRound;
    const V2 = matrix;
    const V1 = V2.slice(0, matrix.length - 1);

    const C1 = helper.getCommitment(V1, participantAddresses, BigInt(config.salt));
    const C2 = helper.getCommitment(V2, participantAddresses, BigInt(config.salt));

    const sig_C1 = helper.sign(config.privateKey, C1);
    const sig_C2 = helper.sign(config.privateKey, C2);

    const counterProver = this.createProver(
      "counter",
      config.maxRoundCount,
      config.maxParticipantCount,
      config.chunkSize,
      config.payoutBatchSize,
      this.counterWasmPath,
      this.counterZkeyPath
    );

    // Measure proving time
    const provingStartTime = Date.now();
    const counterProof = await counterProver.getProof(
      counterRound - 2,
      V2,
      participantAddresses,
      participantAddresses.length,
      [sig_C1.R8x, sig_C1.R8y, sig_C1.S],
      [sig_C2.R8x, sig_C2.R8y, sig_C2.S],
      [publicKey.Ax, publicKey.Ay],
      BigInt(config.salt)
    );
    const provingTime = Date.now() - provingStartTime;

    const counterProofStruct = {
      a: counterProof.a,
      b: counterProof.b,
      c: counterProof.c,
    };

    const counterTx = await flzkpReward
      .connect(owner)
      .counter(1, counterRound, counterProofStruct, { gasPrice: config.gasPrice });
    await counterTx.wait();

    return {
      counterData: { V2, sig_C2 },
      proof: counterProof,
      provingTime,
    };
  }

  async executeRedeemOneShot(config, flzkpReward, owner, helper, publicKey, matrix, participantAddresses, round_number) {
    const V = matrix;
    const salt = BigInt(config.salt);

    const C = helper.getCommitment(V, participantAddresses, salt);
    const sig_C = helper.sign(config.privateKey, C);

    const redeemOneShotProver = this.createProver(
      "redeemOneShot",
      config.maxRoundCount,
      config.maxParticipantCount,
      config.chunkSize,
      config.payoutBatchSize,
      this.redeemOneShotWasmPath,
      this.redeemOneShotZkeyPath
    );

    // Measure proving time
    const provingStartTime = Date.now();
    const redeemOneShotProof = await redeemOneShotProver.getProof(
      round_number - 1,
      V,
      [sig_C.R8x, sig_C.R8y, sig_C.S],
      [publicKey.Ax, publicKey.Ay],
      salt,
      participantAddresses
    );
    const provingTime = Date.now() - provingStartTime;

    const redeemOneShotProofStruct = {
      a: redeemOneShotProof.a,
      b: redeemOneShotProof.b,
      c: redeemOneShotProof.c,
    };

    const rewards = new Array(participantAddresses.length).fill(0n);
    for (let i = 0; i < participantAddresses.length; i++) {
      rewards[i] = BigInt(redeemOneShotProof.S[i]);
    }

    let redeemStatus = "";
    try {
      const redeemOneShotTx = await flzkpReward
        .connect(owner)
        .reward_pay(1, participantAddresses, rewards, redeemOneShotProofStruct, { gasPrice: config.gasPrice });
      await redeemOneShotTx.wait();
      redeemStatus = "Success";
    } catch (error) {
      if (error.message.includes("Too early")) {
        redeemStatus = "Too early";
      } else if (error.message.includes("Not ready")) {
        redeemStatus = "Not ready";
      } else {
        console.error("Error redeeming one shot:", error);
        redeemStatus = "Error: " + error.message;
        throw error;
      }
    }

    return {
      redeemOneShotGas: 0, // We don't have gas info in real chain test
      provingTime,
      publicSignals: redeemOneShotProof.publicSignals,
      S: redeemOneShotProof.S,
      redeemStatus,
    };
  }

  async executeRedeemMultiShot(
    config,
    flzkpReward,
    owner,
    helper,
    publicKey,
    matrix,
    participantAddresses,
    round_number,
    n_shots,
    payoutBatchSize
  ) {
    const V = matrix;
    const salt = BigInt(config.salt);

    const C = helper.getCommitment(V, participantAddresses, salt);
    const sig_C = helper.sign(config.privateKey, C);

    const redeemMultiShotProver = this.createProver(
      "redeemMultiShot",
      config.maxRoundCount,
      config.maxParticipantCount,
      config.chunkSize,
      payoutBatchSize,
      this.redeemMultiShotWasmPath,
      this.redeemMultiShotZkeyPath
    );

    // Measure proving time
    const provingStartTime = Date.now();
    const redeemMultiShotProof = await redeemMultiShotProver.getProof(
      round_number - 1,
      n_shots,
      V,
      [sig_C.R8x, sig_C.R8y, sig_C.S],
      [publicKey.Ax, publicKey.Ay],
      salt,
      participantAddresses
    );
    const provingTime = Date.now() - provingStartTime;

    const redeemMultiShotProofStruct = {
      a: redeemMultiShotProof.a,
      b: redeemMultiShotProof.b,
      c: redeemMultiShotProof.c,
    };

    const remain_number = participantAddresses.length - Number(n_shots) * payoutBatchSize;
    const pnumber = remain_number >= payoutBatchSize ? payoutBatchSize : remain_number % payoutBatchSize;
    const rewards = pnumber > 0 ? new Array(pnumber).fill(0n) : [];
    for (let i = 0; i < pnumber; i++) {
      rewards[i] = BigInt(redeemMultiShotProof.batch_S[i]);
    }
    const participants = pnumber > 0 ? new Array(pnumber).fill(0) : [];
    for (let i = 0; i < pnumber; i++) {
      // Convert BigInt to 32-byte hex string format expected by toAddress
      const hex32 = "0x" + BigInt(redeemMultiShotProof.batch_P[i]).toString(16).padStart(64, "0");
      participants[i] = this.testUtils.toAddress(hex32);
    }

    let redeemStatus = "";
    try {
      const redeemMultiShotTx = await flzkpReward
        .connect(owner)
        .reward_pay_multi(1, participants, rewards, redeemMultiShotProofStruct, { gasPrice: config.gasPrice });
      await redeemMultiShotTx.wait();
      redeemStatus = "Success";
    } catch (error) {
      if (error.message.includes("Too early")) {
        redeemStatus = "Too early";
      } else if (error.message.includes("Not ready")) {
        redeemStatus = "Not ready";
      } else {
        console.error("Error redeeming multi shot:", error);
        redeemStatus = "Error: " + error.message;
        throw error;
      }
    }

    return {
      redeemMultiShotGas: 0, // We don't have gas info in real chain test
      provingTime,
      publicSignals: redeemMultiShotProof.publicSignals,
      batch_S: redeemMultiShotProof.batch_S,
      batch_P: redeemMultiShotProof.batch_P,
      redeemStatus,
    };
  }
}

module.exports = {
  RealChainUtils,
  JobStatus,
  generateWallets,
};

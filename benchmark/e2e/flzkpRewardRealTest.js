require("dotenv").config();
const { ethers } = require("ethers");
const fs = require("fs");
const path = require("path");

BigInt.prototype.toJSON = function () {
  return this.toString();
};

// Import utilities from the real chain utils
const { RealChainUtils, JobStatus, generateWallets } = require("./realChainUtils");

// Import dynamic circuit compilation utilities
// ContractDeployer not needed in disclosure (pre-compiled artifacts)
const { ContractDeploymentService } = require("../../test/utils/contractDeploymentService");
const { generateCircuitId, getDirs } = require("../../test/utils/circuitPaths");
const { ensureCircuitsCompiled } = require("../utils/compilationHelper");

const DEFAULT_CONFIG = {
  participantCount: 5,
  salt: 12345,
  totalReward: ethers.parseEther("0.01"),
  rewardDelay: 3700,
  maxRoundCount: 20,
  maxParticipantCount: 20,
  chunkSize: 16,
  commitRounds: 5,
  maxRewardValue: 10,
  payoutBatchSize: 50,
  localBatchSize: 32,
  useParallelTraining: true, // Enable parallel training by default
  csvFileName: "end2end_real_test_results.csv", // Configurable CSV filename
  gasPrice: ethers.parseUnits("0.001", "gwei"), // Default gas price
  epochs: 2,
  learningRate: 0.001,
  maxParallelWorkersPerGpu: 50, // Max number of Python processes to run in parallel per GPU
  maxRetries: 10, // Max number of retries for failed training tasks
  dataset: "mnist", // 'mnist', 'cifar10', 'fashion-mnist', 'bank-marketing', 'cora', or 'sent140'
  model: "lenet5", // 'lenet5', 'resnet18', 'resnet20', 'resnet50', 'simple-cnn', 'mlp', 'gcn', 'textcnn', or 'fasttext'
  maxDataSizePerParticipant: 60000,
  usePythonTrainer: false, // Switch between JS and Python trainers
  logLevel: "none", // 'none', 'debug', 'info'
  commitMode: false, // Enable ZK-proof-verified commits (disables challenge/counter)
};

// Parse command line arguments
function parseArguments() {
  const args = process.argv.slice(2);
  const options = {
    mode: "full", // 'full', 'redeem-only'
    configFile: null,
    maxParticipants: null,
    maxRounds: null,
    participants: null,
    commitRounds: null,
    chunkSize: null,
    useDefaultCircuits: false,
    useSequentialTraining: false, // Default to parallel training
    csvFileName: null, // CSV filename override
    dataset: DEFAULT_CONFIG.dataset,
    model: DEFAULT_CONFIG.model,
    maxDataSizePerParticipant: null,
    usePythonTrainer: DEFAULT_CONFIG.usePythonTrainer,
    epochs: null,
    learningRate: null,
    commitMode: false, // Default: non-commit mode
    logLevel: null,
    ownerPrivateKey: null,
    challengerPrivateKey: null,
    participantPrivateKeys: null,
    rpcUrl: null,
    network: null,
    gasPrice: null,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--redeem-only":
        options.mode = "redeem-only";
        break;
      case "--config-file":
        options.configFile = args[++i];
        break;
      case "--max-participants":
        options.maxParticipants = parseInt(args[++i]);
        break;
      case "--max-rounds":
        options.maxRounds = parseInt(args[++i]);
        break;
      case "--participants":
        options.participants = parseInt(args[++i]);
        break;
      case "--payout-batch-size":
        options.payoutBatchSize = parseInt(args[++i]);
        break;
      case "--local-batch-size":
        options.localBatchSize = parseInt(args[++i]);
        break;
      case "--rounds":
        options.commitRounds = parseInt(args[++i]);
        break;
      case "--chunk-size":
        options.chunkSize = parseInt(args[++i]);
        break;
      case "--use-default-circuits":
        options.useDefaultCircuits = true;
        break;
      case "--sequential-training":
        options.useSequentialTraining = true;
        break;
      case "--csv-file":
        options.csvFileName = args[++i];
        break;
      case "--dataset":
        options.dataset = args[++i];
        break;
      case "--max-data-size":
        options.maxDataSizePerParticipant = parseInt(args[++i], 10);
        break;
      case "--use-python-trainer":
        options.usePythonTrainer = true;
        break;
      case "--model":
        options.model = args[++i];
        break;
      case "--epochs":
        options.epochs = parseInt(args[++i]);
        break;
      case "--learning-rate":
        options.learningRate = parseFloat(args[++i]);
        break;
      case "--commit-mode":
        options.commitMode = true;
        break;
      case "--log-level":
        options.logLevel = args[++i];
        break;
      case "--owner-private-key":
        options.ownerPrivateKey = args[++i];
        break;
      case "--challenger-private-key":
        options.challengerPrivateKey = args[++i];
        break;
      case "--participant-private-keys":
        options.participantPrivateKeys = args[++i];
        break;
      case "--rpc-url":
        options.rpcUrl = args[++i];
        break;
      case "--network":
        options.network = args[++i];
        break;
      case "--gas-price":
        options.gasPrice = args[++i];
        break;
      case "--help":
        console.log(`
Usage: npm run test:real [options]

Options:
  --redeem-only               Run only redeem process (requires --config-file)
  --config-file <file>        Load/save configuration file
  --max-participants <number> Maximum number of participants for dynamic circuits
  --max-rounds <number>       Maximum number of rounds for dynamic circuits
  --participants <number>     Number of participants for dynamic circuits
  --rounds <number>           Number of rounds for dynamic circuits
  --chunk-size <number>       Chunk size for dynamic circuits
  --use-default-circuits      Use default circuits instead of dynamic compilation
  --sequential-training       Use sequential training instead of parallel training
  --csv-file <file>           CSV output filename (default: benchmark/end2end_real_test_results.csv)
  --dataset <name>            Dataset to use ('mnist', 'cifar10', 'fashion-mnist', 'bank-marketing', 'cora', or 'sent140', default: 'mnist')
  --model <name>              Model to use ('lenet5', 'resnet18', 'resnet20', 'resnet50', 'simple-cnn', 'mlp', 'gcn', 'textcnn', or 'fasttext', default: 'lenet5')
  --max-data-size <number>    Maximum number of data samples per participant
  --use-python-trainer        Use the Python-based federated learning trainer
  --epochs <number>           Number of training epochs (default: 2)
  --learning-rate <number>    Learning rate for training (default: 0.001)
  --commit-mode               Enable commit mode (ZK proof per commit, disables challenge/counter)
  --log-level <level>         Log level ('none', 'debug', 'info', default: 'none')
  --owner-private-key <key>   Owner wallet private key (overrides OWNER_PRIVATE_KEY env var)
  --challenger-private-key <key>  Challenger wallet private key (overrides CHALLENGER_PRIVATE_KEY env var)
  --participant-private-keys <keys>  Comma-separated participant private keys (overrides PARTICIPANT_PRIVATE_KEYS env var)
  --rpc-url <url>             RPC URL (overrides RPC_URL env var, default: http://127.0.0.1:8545)
  --network <name>            Network name (overrides NETWORK env var, default: hardhat)
  --gas-price <gwei>          Gas price in gwei (default: 0.39)
  --help                      Show this help message

Examples:
  npm run test:real                                   # Full process with default circuits
  npm run test:real --participants 5 --rounds 10      # Full process with dynamic circuits
  npm run test:real --dataset cifar10 --model resnet18  # Full process with custom dataset and model
  npm run test:real --epochs 5 --learning-rate 0.01   # Custom training parameters
  npm run test:real --config-file test_config.json    # Full process, save config
  npm run test:real --redeem-only --config-file test_config.json  # Redeem only
  npm run test:real --csv-file my_results.csv         # Custom CSV filename
  npm run test:real --commit-mode                     # Enable commit mode
        `);
        process.exit(0);
    }
  }

  return options;
}

class RealChainTest {
  constructor(options = {}) {
    this.options = options;
    this.config = { ...DEFAULT_CONFIG };
    this.provider = null;
    this.ownerWallet = null;
    this.challengerWallet = null;
    this.realChainUtils = null;
    this.flzkpReward = null;
    this.participantAddresses = [];
    this.savedConfig = null;
    this.circuitId = null;
    this.dirs = null;
    this.verifiers = {};
  }

  async initialize() {
    console.log("🚀 Initializing Real Chain Test...");

    // Load saved config if in redeem-only mode
    if (this.options.mode === "redeem-only" && this.options.configFile) {
      await this.loadSavedConfig();
      return;
    }

    // Update config with command line parameters if provided
    if (this.options.maxParticipants) {
      this.config.maxParticipantCount = this.options.maxParticipants;
    }
    if (this.options.participants) {
      this.config.participantCount = this.options.participants;
    }
    if (this.options.maxRounds) {
      this.config.maxRoundCount = this.options.maxRounds;
    }
    if (this.options.commitRounds) {
      this.config.commitRounds = this.options.commitRounds;
    }
    if (this.options.chunkSize) {
      this.config.chunkSize = this.options.chunkSize;
    }
    if (this.options.payoutBatchSize) {
      this.config.payoutBatchSize = this.options.payoutBatchSize;
    }
    if (this.options.localBatchSize) {
      this.config.localBatchSize = this.options.localBatchSize;
    }
    if (this.options.useSequentialTraining) {
      this.config.useParallelTraining = false;
    }
    if (this.options.csvFileName) {
      this.config.csvFileName = this.options.csvFileName;
    }
    if (this.options.dataset) {
      this.config.dataset = this.options.dataset;
    }
    if (this.options.model) {
      this.config.model = this.options.model;
    }
    if (this.options.epochs !== null) {
      this.config.epochs = this.options.epochs;
    }
    if (this.options.learningRate !== null) {
      this.config.learningRate = this.options.learningRate;
    }
    if (this.options.commitMode) {
      this.config.commitMode = true;
    }
    if (this.options.logLevel !== null) {
      this.config.logLevel = this.options.logLevel;
    }
    if (this.options.gasPrice !== null) {
      this.config.gasPrice = ethers.parseUnits(this.options.gasPrice, "gwei");
    }

    // Auto-extend max values if actual counts exceed them
    if (this.config.participantCount > this.config.maxParticipantCount) {
      this.config.maxParticipantCount = this.config.participantCount;
    }
    if (this.config.commitRounds > this.config.maxRoundCount) {
      this.config.maxRoundCount = this.config.commitRounds;
    }

    // Generate circuit ID and directories for dynamic circuits
    if (!this.options.useDefaultCircuits) {
      this.circuitId = generateCircuitId(
        this.config.maxParticipantCount,
        this.config.maxRoundCount,
        this.config.chunkSize,
        this.config.payoutBatchSize
      );
      this.dirs = getDirs(this.circuitId);
      console.log(`🔧 Using dynamic circuits with ID: ${this.circuitId}`);
    } else {
      console.log(`🔧 Using default circuits`);
    }

    // Load environment variables (CLI flags take precedence over env vars)
    const network = this.options.network || process.env.NETWORK || "hardhat";
    const networkEnvKey = `RPC_URL_${network}`;
    const rpcUrl = this.options.rpcUrl || process.env[networkEnvKey] || process.env.RPC_URL || "http://127.0.0.1:8545";
    const ownerPrivateKey =
      this.options.ownerPrivateKey || process.env[`OWNER_PRIVATE_KEY_${network}`] || process.env.OWNER_PRIVATE_KEY;
    const challengerPrivateKey =
      this.options.challengerPrivateKey || process.env[`CHALLENGER_PRIVATE_KEY_${network}`] || process.env.CHALLENGER_PRIVATE_KEY;
    const participantPrivateKeysString =
      this.options.participantPrivateKeys ||
      process.env[`PARTICIPANT_PRIVATE_KEYS_${network}`] ||
      process.env.PARTICIPANT_PRIVATE_KEYS;

    if (!ownerPrivateKey) {
      throw new Error("OWNER_PRIVATE_KEY not found in .env file");
    }
    this.config.privateKey = ownerPrivateKey;
    if (!challengerPrivateKey) {
      throw new Error("CHALLENGER_PRIVATE_KEY not found in .env file");
    }

    let participantPrivateKeys = [];
    if (participantPrivateKeysString) {
      participantPrivateKeys = participantPrivateKeysString.split(",");
    }

    // Store network for later use
    this.network = network;

    // Redact sensitive parts of RPC URL for logging
    const redactedRpcUrl = rpcUrl.includes("/v2/") ? rpcUrl.replace(/\/v2\/.+$/, "/v2/***") : rpcUrl;
    console.log(`📡 Connecting to ${network} network at ${redactedRpcUrl}`);

    // Setup provider and wallet
    this.provider = new ethers.JsonRpcProvider(rpcUrl);
    this.provider.pollingInterval = 250;
    this.ownerWallet = new ethers.Wallet(ownerPrivateKey, this.provider);
    this.challengerWallet = new ethers.Wallet(challengerPrivateKey, this.provider);

    // show balance of owner and challenger wallets
    // Get explorer URL based on network
    const getExplorerUrl = (address) => {
      if (network === "sepolia") {
        return `https://sepolia.etherscan.io/address/${address}`;
      }
      return address; // For hardhat or other networks
    };

    console.log(
      `💰 Owner balance: ${ethers.formatEther(await this.provider.getBalance(this.ownerWallet.address))} ETH (${getExplorerUrl(
        this.ownerWallet.address
      )})`
    );
    console.log(
      `💰 Challenger balance: ${ethers.formatEther(
        await this.provider.getBalance(this.challengerWallet.address)
      )} ETH (${getExplorerUrl(this.challengerWallet.address)})`
    );

    // Generate participant addresses with dynamic key management
    if (participantPrivateKeys.length > 0) {
      console.log(`📋 Found ${participantPrivateKeys.length} participant private keys in configuration`);
      console.log(`🎯 Requested participant count: ${this.config.participantCount}`);

      if (this.config.participantCount > participantPrivateKeys.length) {
        // Need more keys - generate additional ones
        const additionalKeysNeeded = this.config.participantCount - participantPrivateKeys.length;
        console.log(`➕ Generating ${additionalKeysNeeded} additional participant private keys...`);

        const additionalWallets = generateWallets(additionalKeysNeeded);
        const additionalPrivateKeys = additionalWallets.map((wallet) => wallet.privateKey);

        // Combine existing and new keys
        const allPrivateKeys = [...participantPrivateKeys, ...additionalPrivateKeys];
        this.participantWallets = allPrivateKeys.map((privateKey) => new ethers.Wallet(privateKey, this.provider));

        console.log(`✅ Total participant private keys: ${allPrivateKeys.length}`);
        console.log(`👥 All participant private keys: ${allPrivateKeys.join(",")}`);

        // Stop execution after generating keys
        console.log(`\n🛑 Stopping execution after generating additional participant keys.`);
        console.log(`📝 Please update your .env file with the new PARTICIPANT_PRIVATE_KEYS and run again.`);
        process.exit(0);
      } else if (this.config.participantCount < participantPrivateKeys.length) {
        // Use only the requested number of keys
        console.log(
          `✂️ Using only first ${this.config.participantCount} participant private keys (${participantPrivateKeys.length} available)`
        );
        const selectedPrivateKeys = participantPrivateKeys.slice(0, this.config.participantCount);
        this.participantWallets = selectedPrivateKeys.map((privateKey) => new ethers.Wallet(privateKey, this.provider));
      } else {
        // Exact match - use all keys
        console.log(`✅ Using all ${participantPrivateKeys.length} participant private keys`);
        this.participantWallets = participantPrivateKeys.map((privateKey) => new ethers.Wallet(privateKey, this.provider));
      }
    } else {
      // No keys provided - generate new ones
      console.log(`🆕 No participant private keys found, generating ${this.config.participantCount} new ones...`);
      this.participantWallets = generateWallets(this.config.participantCount);
      const privateKeys = this.participantWallets.map((wallet) => wallet.privateKey);
      console.log(`👥 Participant private keys: ${privateKeys.join(",")}`);
    }

    this.participantAddresses = this.participantWallets.map((wallet) => wallet.address);
    this.config.participantCount = this.participantWallets.length;

    console.log(`👤 Owner address: ${this.ownerWallet.address}`);
    console.log(`👤 Challenger address: ${this.challengerWallet.address}`);
    console.log(`👥 Participant addresses: ${this.participantAddresses.join(", ")}`);

    // Initialize real chain utils
    this.realChainUtils = new RealChainUtils({
      challengeWasmPath: path.join(this.dirs.challenge.dir, "circuit_js", "circuit.wasm"),
      challengeZkeyPath: path.join(this.dirs.challenge.dir, "circuit_final.zkey"),
      counterWasmPath: path.join(this.dirs.counter.dir, "circuit_js", "circuit.wasm"),
      counterZkeyPath: path.join(this.dirs.counter.dir, "circuit_final.zkey"),
      redeemOneShotWasmPath: path.join(this.dirs.redeemOneShot.dir, "circuit_js", "circuit.wasm"),
      redeemOneShotZkeyPath: path.join(this.dirs.redeemOneShot.dir, "circuit_final.zkey"),
      redeemMultiShotWasmPath: path.join(this.dirs.redeemMultiShot.dir, "circuit_js", "circuit.wasm"),
      redeemMultiShotZkeyPath: path.join(this.dirs.redeemMultiShot.dir, "circuit_final.zkey"),
      commitWasmPath: path.join(this.dirs.commit.dir, "circuit_js", "circuit.wasm"),
      commitZkeyPath: path.join(this.dirs.commit.dir, "circuit_final.zkey"),
    });
    await this.realChainUtils.initialize();

    console.log("✅ Initialization complete");
  }

  async loadSavedConfig() {
    console.log(`📂 Loading saved configuration from ${this.options.configFile}`);

    try {
      const configData = JSON.parse(fs.readFileSync(this.options.configFile, "utf8"));
      this.savedConfig = configData;

      // Restore configuration
      this.config = { ...DEFAULT_CONFIG, ...configData.config };
      this.participantAddresses = configData.participantAddresses;

      // Restore circuit information if available
      if (configData.circuitInfo) {
        this.circuitId = configData.circuitInfo.circuitId;
        this.options.useDefaultCircuits = configData.circuitInfo.useDefaultCircuits;
        if (!this.options.useDefaultCircuits) {
          this.dirs = getDirs(this.circuitId);
        }
        console.log(`🔧 Restored circuit info: ${this.circuitId} (default: ${this.options.useDefaultCircuits})`);
      }

      // Restore network from saved config or CLI option
      this.network = configData.network || this.options.network || process.env.NETWORK || "hardhat";

      // Resolve RPC URL: CLI > network-specific env > generic env > default
      const rpcUrl =
        this.options.rpcUrl ||
        process.env[`RPC_URL_${this.network}`] ||
        process.env.RPC_URL ||
        (this.network === "hardhat" ? "http://127.0.0.1:8545" : null);
      if (!rpcUrl) {
        throw new Error(`No RPC URL found for network '${this.network}'. Set RPC_URL_${this.network} or RPC_URL in .env`);
      }
      this.provider = new ethers.JsonRpcProvider(rpcUrl);
      this.provider.pollingInterval = 250;

      // Resolve private keys: CLI > network-specific env > generic env
      const ownerPrivateKey =
        this.options.ownerPrivateKey || process.env[`OWNER_PRIVATE_KEY_${this.network}`] || process.env.OWNER_PRIVATE_KEY;
      const challengerPrivateKey =
        this.options.challengerPrivateKey ||
        process.env[`CHALLENGER_PRIVATE_KEY_${this.network}`] ||
        process.env.CHALLENGER_PRIVATE_KEY;

      if (!ownerPrivateKey || !challengerPrivateKey) {
        throw new Error(
          `OWNER_PRIVATE_KEY and CHALLENGER_PRIVATE_KEY must be set for network '${this.network}' in .env for redeem-only mode`
        );
      }

      this.config.privateKey = ownerPrivateKey;
      this.ownerWallet = new ethers.Wallet(ownerPrivateKey, this.provider);
      this.challengerWallet = new ethers.Wallet(challengerPrivateKey, this.provider);

      // Initialize real chain utils with correct circuit paths
      if (!this.options.useDefaultCircuits && this.dirs) {
        // Use dynamic circuit paths
        this.realChainUtils = new RealChainUtils({
          challengeWasmPath: path.join(this.dirs.challenge.dir, "circuit_js", "circuit.wasm"),
          challengeZkeyPath: path.join(this.dirs.challenge.dir, "circuit_final.zkey"),
          counterWasmPath: path.join(this.dirs.counter.dir, "circuit_js", "circuit.wasm"),
          counterZkeyPath: path.join(this.dirs.counter.dir, "circuit_final.zkey"),
          redeemOneShotWasmPath: path.join(this.dirs.redeemOneShot.dir, "circuit_js", "circuit.wasm"),
          redeemOneShotZkeyPath: path.join(this.dirs.redeemOneShot.dir, "circuit_final.zkey"),
          redeemMultiShotWasmPath: path.join(this.dirs.redeemMultiShot.dir, "circuit_js", "circuit.wasm"),
          redeemMultiShotZkeyPath: path.join(this.dirs.redeemMultiShot.dir, "circuit_final.zkey"),
          commitWasmPath: path.join(this.dirs.commit.dir, "circuit_js", "circuit.wasm"),
          commitZkeyPath: path.join(this.dirs.commit.dir, "circuit_final.zkey"),
        });
      } else {
        // Use default circuits
        this.realChainUtils = new RealChainUtils();
      }
      await this.realChainUtils.initialize();

      console.log("✅ Saved configuration loaded successfully");
      console.log(`📋 Job ID: ${configData.jobId}`);
      console.log(`📊 Current Round: ${configData.currentRound}`);
      console.log(`💰 Contract Address: ${configData.contractAddress}`);
    } catch (error) {
      throw new Error(`Failed to load saved configuration: ${error.message}`);
    }
  }

  async saveConfig(jobResult) {
    if (!this.options.configFile) return;

    console.log(`💾 Saving configuration to ${this.options.configFile}`);

    const configData = {
      timestamp: new Date().toISOString(),
      network: this.network,
      config: {
        participantCount: this.config.participantCount,
        salt: this.config.salt,
        totalReward: this.config.totalReward.toString(),
        rewardDelay: this.config.rewardDelay,
        maxRoundCount: this.config.maxRoundCount,
        maxParticipantCount: this.config.maxParticipantCount,
        chunkSize: this.config.chunkSize,
        commitRounds: this.config.commitRounds,
        maxRewardValue: this.config.maxRewardValue,
        payoutBatchSize: this.config.payoutBatchSize,
        localBatchSize: this.config.localBatchSize,
        useParallelTraining: this.config.useParallelTraining,
        commitMode: this.config.commitMode,
        dataset: this.config.dataset,
        model: this.config.model,
        epochs: this.config.epochs,
        learningRate: this.config.learningRate,
        gasPrice: this.config.gasPrice.toString(),
        logLevel: this.config.logLevel,
      },
      participantAddresses: this.participantAddresses,
      jobId: jobResult.jobId,
      currentRound: jobResult.currentRound,
      contractAddress: await this.flzkpReward.getAddress(),
      matrix: jobResult.matrix.map((row) => row.map((val) => val.toString())),
      rewards: jobResult.rewards.map((val) => val.toString()),
      publicKey: {
        Ax: jobResult.publicKey.Ax.toString(),
        Ay: jobResult.publicKey.Ay.toString(),
      },
      circuitInfo: {
        circuitId: this.circuitId,
        useDefaultCircuits: this.options.useDefaultCircuits,
        participants: this.config.maxParticipantCount,
        rounds: this.config.maxRoundCount,
        chunkSize: this.config.chunkSize,
      },
    };

    // Ensure directory exists
    const dir = path.dirname(this.options.configFile);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(this.options.configFile, JSON.stringify(configData, null, 2));
    console.log("✅ Configuration saved successfully");
  }

  async deployContracts() {
    // Skip deployment if in redeem-only mode
    if (this.options.mode === "redeem-only") {
      console.log("\n📦 Loading existing contract...");
      this.flzkpReward = new ethers.Contract(
        this.savedConfig.contractAddress,
        JSON.parse(fs.readFileSync(path.join(__dirname, "../../artifacts/src/flzkp.sol/FLZKPReward.json"))).abi,
        this.ownerWallet
      );
      console.log(`✅ Contract loaded at: ${this.savedConfig.contractAddress}`);
      return;
    }

    console.log("\n📦 Deploying contracts...");

    if (this.options.useDefaultCircuits) {
      // Use default circuits (original behavior)
      return await this.deployDefaultContracts();
    } else {
      // Use dynamic circuits
      return await this.deployDynamicContracts();
    }
  }

  async deployDefaultContracts() {
    console.log("  Using default circuits...");

    const shouldSkipRedeemOneShot = this.config.maxParticipantCount > 50;

    const service = new ContractDeploymentService({
      useDynamicCircuits: false,
      signer: this.ownerWallet,
      deployOptions: { gasPrice: this.config.gasPrice },
      network: this.network,
    });

    const result = await service.deployContracts(ethers, {
      skipRedeemOneShot: shouldSkipRedeemOneShot,
      skipCommit: !this.config.commitMode,
    });
    this.flzkpReward = result.flzkpReward;
    this.verifiers = result.verifiers;

    return result;
  }

  async deployDynamicContracts() {
    console.log("  Using dynamic circuits...");

    // Compile circuits if needed
    await this.compileCircuitsIfNeeded();

    const shouldSkipRedeemOneShot = this.config.maxParticipantCount > 50;

    const service = new ContractDeploymentService({
      useDynamicCircuits: true,
      circuitId: this.circuitId,
      baseDir: this.dirs.baseDir,
      participants: this.config.maxParticipantCount,
      rounds: this.config.maxRoundCount,
      batchSize: this.config.payoutBatchSize,
      signer: this.ownerWallet,
      deployOptions: { gasPrice: this.config.gasPrice },
      network: this.network,
    });

    const result = await service.deployContracts(ethers, {
      skipRedeemOneShot: shouldSkipRedeemOneShot,
      skipCommit: !this.config.commitMode,
    });
    this.flzkpReward = result.flzkpReward;
    this.verifiers = result.verifiers;

    return result;
  }

  async compileCircuitsIfNeeded() {
    console.log("  Checking circuit compilation status...");

    // Skip redeemOneShot if participant count > 50 (will use multi-shot instead)
    const shouldSkipRedeemOneShot = this.config.maxParticipantCount > 50;
    let circuitTypes = ["challenge", "counter", "redeemMultiShot"];

    if (this.config.commitMode) {
      circuitTypes.unshift("commit");
    } else {
      console.log("  ⚡ Skipping commit circuit (non-commit mode)");
    }

    if (!shouldSkipRedeemOneShot) {
      circuitTypes.splice(circuitTypes.indexOf("redeemMultiShot"), 0, "redeemOneShot");
    } else {
      console.log("  ⚡ Skipping redeemOneShot circuit (participant count > 50, will use multi-shot)");
    }

    // Ensure all circuits are compiled (check existing or compile if needed)
    // Map config to expected format for ensureCircuitsCompiled
    const compileConfig = {
      participants: this.config.maxParticipantCount,
      rounds: this.config.maxRoundCount,
      chunkSize: this.config.chunkSize,
      batchSize: this.config.payoutBatchSize,
    };

    const { circuitResults, compileTimes } = await ensureCircuitsCompiled(circuitTypes, this.circuitId, compileConfig);

    console.log("  Circuit compilation completed successfully");
  }

  async setupJob() {
    // Skip setup if in redeem-only mode
    if (this.options.mode === "redeem-only") {
      console.log("\n🔧 Skipping job setup (redeem-only mode)...");

      // Recreate helper for redeem-only mode
      const { RewardMatrixHelper } = require("../../test/utils/RewardMatrixHelper");
      const helper = new RewardMatrixHelper(this.config.maxRoundCount, this.config.maxParticipantCount, this.config.chunkSize);
      await helper.init();

      return {
        jobId: this.savedConfig.jobId,
        currentRound: this.savedConfig.currentRound,
        matrix: this.savedConfig.matrix.map((row) => row.map((val) => BigInt(val))),
        rewards: this.savedConfig.rewards.map((val) => BigInt(val)),
        publicKey: {
          Ax: BigInt(this.savedConfig.publicKey.Ax),
          Ay: BigInt(this.savedConfig.publicKey.Ay),
        },
        helper: helper,
        trainingHistory: [],
      };
    }

    console.log(`💰 Total reward: ${ethers.formatEther(this.config.totalReward)} ETH`);
    console.log("\n🔧 Setting up job...");

    // Initialize RealChainUtils with correct circuit paths if not already done
    if (!this.options.useDefaultCircuits && !this.realChainUtils) {
      // Use dynamic circuit paths
      this.realChainUtils = new RealChainUtils({
        challengeWasmPath: path.join(this.dirs.challenge.dir, "circuit_js", "circuit.wasm"),
        challengeZkeyPath: path.join(this.dirs.challenge.dir, "circuit_final.zkey"),
        counterWasmPath: path.join(this.dirs.counter.dir, "circuit_js", "circuit.wasm"),
        counterZkeyPath: path.join(this.dirs.counter.dir, "circuit_final.zkey"),
        redeemOneShotWasmPath: path.join(this.dirs.redeemOneShot.dir, "circuit_js", "circuit.wasm"),
        redeemOneShotZkeyPath: path.join(this.dirs.redeemOneShot.dir, "circuit_final.zkey"),
        redeemMultiShotWasmPath: path.join(this.dirs.redeemMultiShot.dir, "circuit_js", "circuit.wasm"),
        redeemMultiShotZkeyPath: path.join(this.dirs.redeemMultiShot.dir, "circuit_final.zkey"),
        commitWasmPath: path.join(this.dirs.commit.dir, "circuit_js", "circuit.wasm"),
        commitZkeyPath: path.join(this.dirs.commit.dir, "circuit_final.zkey"),
      });
      await this.realChainUtils.initialize();
    }

    const jobResult = await this.realChainUtils.setupJob(
      this.config,
      this.flzkpReward,
      this.ownerWallet,
      this.participantAddresses,
      (i) => {
        console.log(`  🔄 Commit round ${i + 1}/${this.config.commitRounds}`);
      }
    );

    console.log(`✅ Job setup complete. Job ID: 1`);
    console.log(`📊 Current round: ${jobResult.currentRound}`);

    // Save configuration if requested
    if (this.options.configFile) {
      await this.saveConfig(jobResult);
    }

    // Display detailed training results
    if (jobResult.trainingHistory) {
      console.log("\n📈 === Detailed Training Results ===");
      jobResult.trainingHistory.forEach((round, index) => {
        console.log(`\nRound ${index + 1}:`);
        console.log(`  Test Accuracy: ${(round.evaluationResult.testAccuracy * 100).toFixed(2)}%`);
        console.log(`  Test Loss: ${round.evaluationResult.testLoss.toFixed(4)}`);
        console.log(`  Participants trained: ${round.participantResults.length}`);

        round.participantResults.forEach((participant, pIdx) => {
          console.log(
            `    Participant ${pIdx + 1}: Accuracy=${(participant.metrics.accuracy * 100).toFixed(
              2
            )}%, Loss=${participant.metrics.loss.toFixed(4)}`
          );
        });
      });
    }

    return jobResult;
  }

  async executeChallenge(jobResult) {
    // Log balances after commits and before challenge
    console.log(
      `💰 Owner balance after commits: ${ethers.formatEther(await this.provider.getBalance(this.ownerWallet.address))} ETH`
    );
    console.log(
      `💰 Contract balance after commits: ${ethers.formatEther(
        await this.provider.getBalance(await this.flzkpReward.getAddress())
      )} ETH`
    );

    console.log("\n⚔️ Executing challenge...");

    const challengeResult = await this.realChainUtils.executeChallenge(
      this.config,
      this.flzkpReward,
      this.challengerWallet,
      jobResult.helper,
      jobResult.publicKey,
      jobResult.matrix,
      jobResult.currentRound,
      this.participantAddresses,
      jobResult.rewards
    );

    const jobInfo = await this.flzkpReward.jobInfo(1);
    console.log(`✅ Challenge executed. Status: ${jobInfo.status} (${JobStatus[jobInfo.status]})`);
    console.log(`🎯 Challenge round: ${challengeResult.challengeRound}`);

    return challengeResult;
  }

  async executeCounter(jobResult, challengeResult) {
    console.log("\n🛡️ Executing counter...");

    const counterResult = await this.realChainUtils.executeCounter(
      this.config,
      this.flzkpReward,
      this.ownerWallet,
      jobResult.helper,
      jobResult.publicKey,
      jobResult.matrix,
      challengeResult.challengeRound,
      this.participantAddresses
    );

    const jobInfo = await this.flzkpReward.jobInfo(1);
    console.log(`✅ Counter executed. Status: ${jobInfo.status} (${JobStatus[jobInfo.status]})`);

    return counterResult;
  }

  async executeRedeem(jobResult, challengeResult) {
    // Determine if we should use multishot based on participant count vs batch size
    const shouldUseMultiShot = this.config.maxParticipantCount > this.config.payoutBatchSize;

    console.log(`\n💸 Redeem configuration:`);
    console.log(`   Max participants: ${this.config.maxParticipantCount}`);
    console.log(`   Batch size: ${this.config.payoutBatchSize}`);
    console.log(`   Using multi-shot: ${shouldUseMultiShot}`);

    if (shouldUseMultiShot) {
      console.log("\n💸 Executing redeem multi-shot...");
      return await this.executeRedeemMultiShot(jobResult, challengeResult);
    } else {
      console.log("\n💸 Executing redeem one-shot...");
      return await this.executeRedeemOneShot(jobResult, challengeResult);
    }
  }

  async executeRedeemOneShot(jobResult, challengeResult) {
    const redeemResult = await this.realChainUtils.executeRedeemOneShot(
      this.config,
      this.flzkpReward,
      this.ownerWallet,
      jobResult.helper,
      jobResult.publicKey,
      jobResult.matrix,
      this.participantAddresses,
      challengeResult.challengeRound
    );

    if (redeemResult.redeemStatus !== "Success") {
      if (redeemResult.redeemStatus === "Too early") {
        console.log("⚠ Redeem one shot result: Too early");
      } else if (redeemResult.redeemStatus === "Not ready") {
        console.log("⚠ Redeem one shot result: Not ready. Did you already redeem?");
      } else {
        console.log(`❌ Redeem one shot result: ${redeemResult.redeemStatus}`);
        throw new Error("Redeem one shot failed");
      }
    } else {
      const jobInfo = await this.flzkpReward.jobInfo(1);
      console.log(`✅ Redeem one shot executed. Status: ${jobInfo.status} (${JobStatus[jobInfo.status]})`);
    }

    return redeemResult;
  }

  async executeRedeemMultiShot(jobResult, challengeResult) {
    const payoutBatchSize = this.config.payoutBatchSize;
    let shotCount = 0;
    let totalProvingTime = 0;

    // Continue executing multi-shot until all participants are paid
    while (true) {
      try {
        // Get current job info to check status and n_shots
        let jobInfo = await this.flzkpReward.jobInfo(1);

        // If job is already paid, break the loop
        if (jobInfo.status === 4n) {
          // JobStatus.Paid = 4
          console.log(`✅ All participants have been paid. Total shots: ${shotCount}`);
          break;
        }

        const nShots = jobInfo.n_shots;
        shotCount++;

        console.log(`  🔄 Executing multi-shot ${shotCount} (n_shots: ${nShots})...`);

        const redeemMultiShotResult = await this.realChainUtils.executeRedeemMultiShot(
          this.config,
          this.flzkpReward,
          this.ownerWallet,
          jobResult.helper,
          jobResult.publicKey,
          jobResult.matrix,
          this.participantAddresses,
          challengeResult.challengeRound,
          nShots,
          payoutBatchSize
        );

        totalProvingTime += redeemMultiShotResult.provingTime;

        if (redeemMultiShotResult.redeemStatus !== "Success") {
          if (redeemMultiShotResult.redeemStatus === "Too early") {
            console.log(`⚠ Multi-shot ${shotCount} result: Too early. Exiting loop.`);
            break;
          } else if (redeemMultiShotResult.redeemStatus === "Not ready") {
            console.log(`⚠ Multi-shot ${shotCount} result: Not ready. Did you already redeem?`);
            break;
          } else {
            console.log(`❌ Multi-shot ${shotCount} result: ${redeemMultiShotResult.redeemStatus}`);
            throw new Error(`Multi-shot ${shotCount} failed`);
          }
        } else {
          console.log(`  ✅ Multi-shot ${shotCount} executed successfully (${redeemMultiShotResult.provingTime}ms proving time)`);

          // Check if we should continue (job status should be Paying or we need more shots)
          jobInfo = await this.flzkpReward.jobInfo(1);
          if (jobInfo.status === 4n) {
            // JobStatus.Paid = 4
            console.log(`✅ All participants have been paid. Total shots: ${shotCount}`);
            break;
          }
        }
      } catch (shotError) {
        console.error(`❌ Multi-shot ${shotCount + 1} failed:`, shotError.message);
        throw shotError;
      }
    }

    const jobInfo = await this.flzkpReward.jobInfo(1);
    console.log(`✅ Multi-shot redeem completed. Final status: ${jobInfo.status} (${JobStatus[jobInfo.status]})`);
    console.log(`📊 Total shots executed: ${shotCount}, Total proving time: ${totalProvingTime}ms`);

    return {
      redeemMultiShotGas: 0, // We don't have gas info in real chain test
      provingTime: totalProvingTime,
      shotCount,
      redeemStatus: "Success",
    };
  }

  async runTest() {
    const startTime = new Date();
    try {
      console.log("🎯 Starting FLZKPReward Real Chain Test");
      console.log(`⏰ Start time: ${startTime.toISOString()}`);
      console.log(`📋 Mode: ${this.options.mode}`);
      if (this.options.configFile) {
        console.log(`📁 Config file: ${this.options.configFile}`);
      }

      // Initialize
      await this.initialize();

      console.log("=".repeat(50));
      console.log("📋 Configuration:");
      const printableConfig = { ...this.config, privateKey: this.config.privateKey ? "[REDACTED]" : undefined };
      console.log(JSON.stringify(printableConfig, null, 2));
      console.log("=".repeat(50));

      // Deploy contracts
      await this.deployContracts();

      if (this.options.mode === "full") {
        // Full mode: setup → [challenge → counter if not commit mode] → redeem → CSV
        console.log("\n🔄 Running full process...");

        const jobResult = await this.setupJob();

        // In commit mode, challenge and counter are disabled by the contract
        const isCommitMode = await this.flzkpReward.commitMode();
        let challengeResult;
        if (isCommitMode) {
          console.log("\n🔐 Commit mode active — skipping challenge and counter");
          challengeResult = { challengeRound: jobResult.currentRound };
        } else {
          challengeResult = await this.executeChallenge(jobResult);
          await this.executeCounter(jobResult, challengeResult);
        }

        await this.executeRedeem(jobResult, challengeResult);

        // Export CSV data
        if (this.realChainUtils.flTrainer) {
          const csvData = this.realChainUtils.flTrainer.getCSVData();
          const csvContent = this.realChainUtils.flTrainer.exportCSV();

          const csvFileName = this.config.csvFileName;
          if (fs.existsSync(csvFileName)) {
            const dataRows = csvContent.split("\n").slice(1).join("\n");
            fs.appendFileSync(csvFileName, "\n" + dataRows);
          } else {
            fs.writeFileSync(csvFileName, csvContent);
          }

          console.log("\n📊 CSV Results exported to:", csvFileName);
          console.log(`📈 Total CSV entries: ${csvData.length}`);
        }
      } else {
        // Redeem-only mode: only redeem
        console.log("\n💸 Running redeem-only process...");

        const jobResult = await this.setupJob(); // This loads the saved config
        await this.executeRedeem(jobResult, { challengeRound: jobResult.currentRound });
      }

      // Final status check
      const finalJobInfo = await this.flzkpReward.jobInfo(1);

      const endTime = new Date();
      const duration = endTime - startTime;

      console.log("\n🎉 Test completed successfully!");
      console.log(`⏰ End time: ${endTime.toISOString()}`);
      console.log(`⏱️ Total duration: ${duration}ms (${(duration / 1000).toFixed(2)}s)`);
      console.log(`📋 Final job status: ${finalJobInfo.status} (${JobStatus[finalJobInfo.status]})`);
      console.log(
        `💰 Contract balance: ${ethers.formatEther(await this.provider.getBalance(await this.flzkpReward.getAddress()))} ETH`
      );
      console.log(`💰 Owner balance: ${ethers.formatEther(await this.provider.getBalance(this.ownerWallet.address))} ETH`);
      console.log(
        `💰 Challenger balance: ${ethers.formatEther(await this.provider.getBalance(this.challengerWallet.address))} ETH`
      );
      // display the balance of each participant
      for (const participant of this.participantAddresses) {
        console.log(
          `💰 Participant ${participant} balance:\n    ${ethers.formatEther(await this.provider.getBalance(participant))} ETH`
        );
      }
      process.exit(0);
    } catch (error) {
      const endTime = new Date();
      const duration = endTime - startTime;

      console.error("❌ Test failed:", error);
      console.error(`⏰ End time: ${endTime.toISOString()}`);
      console.error(`⏱️ Total duration: ${duration}ms (${(duration / 1000).toFixed(2)}s)`);
      throw error;
    }
  }
}

// Main execution
async function main() {
  const options = parseArguments();
  const test = new RealChainTest(options);
  await test.runTest();
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

module.exports = { RealChainTest };

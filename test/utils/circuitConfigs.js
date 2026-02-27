const path = require("path");

// Circuit configurations
const CIRCUIT_CONFIGS = {
  challenge: {
    name: "Challenge",
    templateName: "ChallengeProof",
    publicParameters: ["round_number"],
    contractName: "ChallengeVerifier",
  },
  counter: {
    name: "Counter",
    templateName: "CounterChallengeProof",
    publicParameters: ["round_number"],
    contractName: "CounterVerifier",
  },
  redeemOneShot: {
    name: "RedeemOneShot",
    templateName: "RedeemOneShot",
    publicParameters: ["P", "round_number"],
    contractName: "RedeemOneShotVerifier",
  },
  redeemMultiShot: {
    name: "RedeemMultiShot",
    templateName: "RedeemMultiShot",
    publicParameters: ["round_number", "n_shots"],
    contractName: "RedeemMultiShotVerifier",
  },
  commit: {
    name: "Commit",
    templateName: "CommitProof",
    publicParameters: ["round_number"],
    contractName: "CommitVerifier",
  },
  redeemKeccak: {
    name: "RedeemKeccak",
    templateName: "RedeemKeccak",
    publicParameters: ["round_number"],
    contractName: "RedeemKeccakVerifier",
  },
};

// Helper function to create circuit paths
function createCircuitPaths(circuitId, circuitName) {
  const basePath = path.join(__dirname, "..", "..");
  return {
    circuitDir: path.join(basePath, "target", "benchmark", circuitId, circuitName),
    libPath: path.join(basePath, "node_modules"),
    includePath: path.join(basePath, "circuits", `${circuitName}.circom`),
  };
}

// Factory function to create circuit-specific compilers
function createCircuitCompiler(circuitType, circuitId) {
  const { CircuitCompiler } = require("./circuitCompiler");
  const baseConfig = CIRCUIT_CONFIGS[circuitType];

  if (!baseConfig) {
    throw new Error(`Unknown circuit type: ${circuitType}`);
  }

  const config = {
    ...baseConfig,
    ...createCircuitPaths(circuitId, baseConfig.name),
  };

  return new CircuitCompiler(config);
}

module.exports = { CIRCUIT_CONFIGS, createCircuitCompiler };

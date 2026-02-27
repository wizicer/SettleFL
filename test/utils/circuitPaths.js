/**
 * Centralized circuit path configuration utilities.
 *
 * This module consolidates all circuit path-related functions that were
 * previously duplicated across multiple files:
 * - test/real/flzkpRewardRealTest.js
 * - benchmark/benchmark.js
 * - benchmark/benchmark_scalability.js
 * - benchmark/benchmark_zkrfl.js
 */

const fs = require("fs");
const path = require("path");

// Default circuit output directory (can be overridden)
const DEFAULT_OBJ_DIR = path.join(__dirname, "..", "..", "target", "benchmark");

/**
 * Generate a unique circuit ID from parameters.
 * @param {number} participants - Number of participants
 * @param {number} rounds - Number of rounds
 * @param {number} chunkSize - Chunk size for circuit
 * @param {number} [batchSize] - Optional batch size for multi-shot circuits
 * @returns {string} Circuit ID string
 */
function generateCircuitId(participants, rounds, chunkSize, batchSize = null) {
  if (batchSize !== null) {
    return `${participants}_${rounds}_${chunkSize}_${batchSize}`;
  }
  return `${participants}_${rounds}_${chunkSize}`;
}

/**
 * Calculate appropriate batch size based on participant count.
 * @param {number} participants - Number of participants
 * @returns {number} Recommended batch size
 */
function getBatchSize(participants) {
  return participants < 10 ? Math.min(participants, 10) : participants <= 50 ? Math.min(participants, 25) : 50;
}

/**
 * Get circuit parameters based on circuit type.
 * @param {string} circuitType - Type of circuit (challenge, counter, redeemOneShot, redeemMultiShot, commit)
 * @param {number} participants - Number of participants
 * @param {number} rounds - Number of rounds
 * @param {number} chunkSize - Chunk size
 * @param {number} [batchSize] - Batch size (required for redeemMultiShot)
 * @returns {string[]} Array of circuit parameters as strings
 */
function getCircuitParameters(circuitType, participants, rounds, chunkSize, batchSize = null) {
  switch (circuitType) {
    case "challenge":
    case "counter":
    case "redeemOneShot":
    case "commit":
      return [participants.toString(), rounds.toString(), chunkSize.toString()];
    case "redeemMultiShot":
      if (batchSize === null) {
        batchSize = getBatchSize(participants);
      }
      return [participants.toString(), rounds.toString(), chunkSize.toString(), batchSize.toString()];
    default:
      throw new Error(`Unknown circuit type: ${circuitType}`);
  }
}

/**
 * Get directory paths for all circuit types.
 * @param {string} circuitId - Circuit ID
 * @param {string} [objDir] - Base object directory (defaults to target/benchmark)
 * @returns {Object} Object containing directory paths for each circuit type
 */
function getDirs(circuitId, objDir = DEFAULT_OBJ_DIR) {
  const baseDir = path.join(objDir, circuitId);
  const challengeDir = path.join(baseDir, "Challenge");
  const counterDir = path.join(baseDir, "Counter");
  const redeemOneShotDir = path.join(baseDir, "RedeemOneShot");
  const redeemMultiShotDir = path.join(baseDir, "RedeemMultiShot");
  const commitDir = path.join(baseDir, "Commit");

  return {
    baseDir,
    challenge: {
      dir: challengeDir,
      metadataPath: path.join(challengeDir, "metadata.json"),
    },
    counter: {
      dir: counterDir,
      metadataPath: path.join(counterDir, "metadata.json"),
    },
    redeemOneShot: {
      dir: redeemOneShotDir,
      metadataPath: path.join(redeemOneShotDir, "metadata.json"),
    },
    redeemMultiShot: {
      dir: redeemMultiShotDir,
      metadataPath: path.join(redeemMultiShotDir, "metadata.json"),
    },
    commit: {
      dir: commitDir,
      metadataPath: path.join(commitDir, "metadata.json"),
    },
  };
}

/**
 * Check if a single circuit type exists with all required files.
 * @param {string} dir - Circuit directory
 * @param {string} metadataPath - Path to metadata file
 * @returns {boolean} True if circuit exists with all required files
 */
function checkCircuitExists(dir, metadataPath) {
  return (
    fs.existsSync(dir) &&
    fs.existsSync(path.join(dir, "circuit_final.zkey")) &&
    fs.existsSync(path.join(dir, "circuit_js", "circuit.wasm")) &&
    fs.existsSync(metadataPath)
  );
}

/**
 * Check if circuits are already compiled with metadata.
 * @param {string} circuitId - Circuit ID
 * @param {string} [objDir] - Base object directory
 * @returns {Object} Object with boolean flags for each circuit type
 */
function checkExistingCircuits(circuitId, objDir = DEFAULT_OBJ_DIR) {
  const dirs = getDirs(circuitId, objDir);

  return {
    challengeExists: checkCircuitExists(dirs.challenge.dir, dirs.challenge.metadataPath),
    counterExists: checkCircuitExists(dirs.counter.dir, dirs.counter.metadataPath),
    redeemOneShotExists: checkCircuitExists(dirs.redeemOneShot.dir, dirs.redeemOneShot.metadataPath),
    redeemMultiShotExists: checkCircuitExists(dirs.redeemMultiShot.dir, dirs.redeemMultiShot.metadataPath),
    commitExists: checkCircuitExists(dirs.commit.dir, dirs.commit.metadataPath),
  };
}

/**
 * Save circuit metadata to file.
 * @param {string} circuitId - Circuit ID
 * @param {string} circuitType - Type of circuit
 * @param {Object} metadata - Metadata object to save
 * @param {string} [objDir] - Base object directory
 */
function saveCircuitMetadata(circuitId, circuitType, metadata, objDir = DEFAULT_OBJ_DIR) {
  const dirs = getDirs(circuitId, objDir);
  const { dir, metadataPath } = dirs[circuitType];

  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const metadataContent = {
    participants: metadata.participants,
    rounds: metadata.rounds,
    chunkSize: metadata.chunkSize,
    batchSize: metadata.batchSize,
    circuitType,
    circuitId,
    constraints: metadata.constraints,
    wires: metadata.wires,
    templateInstances: metadata.templateInstances,
    publicInputs: metadata.publicInputs,
    privateInputs: metadata.privateInputs,
    publicOutputs: metadata.publicOutputs,
    compileTime: metadata.compileTime,
    timestamp: new Date().toISOString(),
  };

  fs.writeFileSync(metadataPath, JSON.stringify(metadataContent, null, 2));
}

/**
 * Read circuit metadata from file.
 * @param {string} circuitId - Circuit ID
 * @param {string} circuitType - Type of circuit
 * @param {string} [objDir] - Base object directory
 * @returns {Object|null} Metadata object or null if not found
 */
function readCircuitMetadata(circuitId, circuitType, objDir = DEFAULT_OBJ_DIR) {
  const dirs = getDirs(circuitId, objDir);
  const { metadataPath } = dirs[circuitType];

  if (!fs.existsSync(metadataPath)) {
    return null;
  }

  try {
    const metadataContent = fs.readFileSync(metadataPath, "utf8");
    return JSON.parse(metadataContent);
  } catch (error) {
    console.log(`  Warning: Could not read metadata from ${metadataPath}: ${error.message}`);
    return null;
  }
}

module.exports = {
  DEFAULT_OBJ_DIR,
  generateCircuitId,
  getBatchSize,
  getCircuitParameters,
  getDirs,
  checkCircuitExists,
  checkExistingCircuits,
  saveCircuitMetadata,
  readCircuitMetadata,
};

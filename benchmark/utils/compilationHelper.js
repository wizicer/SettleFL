const {
  checkExistingCircuits,
  getCircuitParameters,
  getDirs,
  readCircuitMetadata,
} = require("../../test/utils/circuitPaths");

async function ensureCircuitsCompiled(circuitTypes, circuitId, config, options = {}) {
  const existingCircuits = checkExistingCircuits(circuitId);
  const circuitResults = {};
  const compileTimes = {};

  for (const circuitType of circuitTypes) {
    const existingKey = `${circuitType}Exists`;

    if (existingCircuits[existingKey]) {
      console.log(`  ✓ Using pre-compiled ${circuitType} circuit`);
      const metadata = readCircuitMetadata(circuitId, circuitType);
      if (metadata) {
        circuitResults[circuitType] = { success: true, ...metadata };
        compileTimes[circuitType] = metadata.compileTime || 0;
      } else {
        circuitResults[circuitType] = { success: true };
        compileTimes[circuitType] = 0;
      }
    } else {
      throw new Error(
        `Pre-compiled circuit not found for ${circuitType} (circuit ID: ${circuitId}). ` +
        `This disclosure project uses pre-compiled artifacts only.`
      );
    }
  }

  return { circuitResults, compileTimes };
}

module.exports = { ensureCircuitsCompiled };

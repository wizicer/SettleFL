const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

class ContractDeployer {
  constructor(baseDir, circuitId) {
    this.baseDir = baseDir;
    this.circuitId = circuitId;

    // Create temp folder for this specific circuit compilation
    this.contractsDir = path.join(this.baseDir, `contracts`);
    if (!fs.existsSync(this.contractsDir)) {
      fs.mkdirSync(this.contractsDir, { recursive: true });
    }
  }

  capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  async prepareVerifier(circuitType) {
    const circuitName = this.capitalize(circuitType);
    const contractName = `${circuitName}Verifier`;
    const circuitDir = path.join(this.baseDir, `${circuitName}`);

    // Copy verifier from compiled circuit
    const verifierSourcePath = path.join(circuitDir, "verifier.sol");
    const verifierCode = fs.readFileSync(verifierSourcePath, "utf8");

    // Rename contract and add SPDX if needed
    let fullContractCode = verifierCode.replace(/contract Groth16Verifier/g, `contract ${contractName}_${this.circuitId}`);

    if (!fullContractCode.includes("// SPDX-License-Identifier:")) {
      fullContractCode = `// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;

${fullContractCode}`;
    }

    const contractFileName = `${contractName}.sol`;
    const contractPath = path.join(this.contractsDir, contractFileName);
    fs.writeFileSync(contractPath, fullContractCode, "utf8");
  }

  /**
   * Special copy and replace function for FLZKPReward contract
   * @param {number} n_participants - Number of participants
   * @param {number} batch_size - Batch size
   */
  async prepareFLZKPReward(n_participants, batch_size) {
    const sourcePath = path.join(__dirname, "..", "..", "src", "flzkp.sol");
    const targetPath = path.join(this.contractsDir, "FLZKPReward.sol");

    // Read the source file
    let contractCode = fs.readFileSync(sourcePath, "utf8");

    // Perform replacements
    // 1. Replace uint256[23] calldata publicSignals with max*2+3
    const maxPublicSignals = n_participants * 2 + 3;
    contractCode = contractCode.replace(
      /uint256\[23\] calldata publicSignals/g,
      `uint256[${maxPublicSignals}] calldata publicSignals`
    );

    // 2. Replace uint256[14] calldata publicSignals with batch*2+4
    const batchPublicSignals = batch_size * 2 + 4;
    contractCode = contractCode.replace(
      /uint256\[14\] calldata publicSignals/g,
      `uint256[${batchPublicSignals}] calldata publicSignals`
    );

    // 3. Add circuit ID suffix to contract name
    contractCode = contractCode.replace(/contract FLZKPReward/g, `contract FLZKPReward_${this.circuitId}`);

    // 4. Replace MAX_PARTICIPANTS and BATCH_SIZE constants
    contractCode = contractCode.replace(
      /uint256 public constant MAX_PARTICIPANTS = 10;/g,
      `uint256 public constant MAX_PARTICIPANTS = ${n_participants};`
    );

    contractCode = contractCode.replace(
      /uint256 public constant BATCH_SIZE = 5;/g,
      `uint256 public constant BATCH_SIZE = ${batch_size};`
    );

    // Write the modified contract to the target directory
    fs.writeFileSync(targetPath, contractCode, "utf8");
  }

  prepareZKRFL(initialCommitment) {
    const sourcePath = path.join(__dirname, "..", "..", "src", "zkrfl.sol");
    const targetPath = path.join(this.contractsDir, "ZKRFL.sol");
    let contractCode = fs.readFileSync(sourcePath, "utf8");
    // 1. Replace initial commitment
    contractCode = contractCode.replace(
      /4386688658862886042738112122485947876856567881153223404339065030203865120868/g,
      initialCommitment.toString()
    );
    // 2. Add circuit ID suffix to contract name
    contractCode = contractCode.replace(/contract FLZKRollup/g, `contract FLZKRollup_${this.circuitId}`);
    fs.writeFileSync(targetPath, contractCode, "utf8");
  }

  async compileVerifier() {
    hre.config.paths.sources = this.baseDir;
    await hre.run("compile", {});
  }

  async deployVerifier(circuitType, ethers) {
    const circuitName = this.capitalize(circuitType);
    const contractName = `${circuitName}Verifier`;

    // Deploy the compiled verifier contract
    const VerifierFactory = await ethers.getContractFactory(`${contractName}_${this.circuitId}`);
    const verifier = await VerifierFactory.deploy();
    await verifier.waitForDeployment();

    return verifier;
  }

  /**
   * Generates a basic hardhat.config.ts file into the target directory.
   */
  generateHardhatConfig() {
    const dir = this.baseDir;
    const configContent = `require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.28"
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  }
}; 
`;

    const outputPath = path.resolve(dir, "hardhat.config.js");
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(outputPath, configContent);
  }
}

module.exports = { ContractDeployer };

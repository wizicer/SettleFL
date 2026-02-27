const fs = require("fs");
const path = require("path");
const { RewardMatrixHelper } = require("./RewardMatrixHelper");

const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";

class ContractDeploymentService {
  constructor(options = {}) {
    this.useDynamicCircuits = options.useDynamicCircuits || false;
    this.circuitId = options.circuitId;
    this.baseDir = options.baseDir;
    this.participants = options.participants;
    this.rounds = options.rounds;
    this.batchSize = options.batchSize;
    this.deployOptions = options.deployOptions || {};
    this.signer = options.signer;
    this.network = options.network || "hardhat";
  }

  async deployContracts(ethers, options = {}) {
    const { skipRedeemOneShot = false, skipCommit = true } = options;
    return this._deployFromPrecompiledArtifacts(ethers, skipRedeemOneShot, skipCommit);
  }

  async _deployFromPrecompiledArtifacts(ethers, skipRedeemOneShot, skipCommit) {
    console.log("  Deploying from pre-compiled artifacts...");

    const artifactsBase = path.join(
      __dirname, "..", "..", "artifacts", "target", "benchmark", this.circuitId, "contracts"
    );

    const verifiers = {};
    const circuitTypes = [];

    if (!skipCommit) {
      circuitTypes.push("commit");
    } else {
      console.log("  ⚡ CommitVerifier deployment skipped (non-commit mode)");
      verifiers.commit = null;
    }

    circuitTypes.push("challenge", "counter");

    if (!skipRedeemOneShot) {
      circuitTypes.push("redeemOneShot");
    } else {
      verifiers.redeemOneShot = null;
      console.log("  ⚡ RedeemOneShotVerifier deployment skipped");
    }

    circuitTypes.push("redeemMultiShot");

    for (const circuitType of circuitTypes) {
      const circuitName = circuitType.charAt(0).toUpperCase() + circuitType.slice(1);
      const contractName = `${circuitName}Verifier_${this.circuitId}`;
      const artifactPath = path.join(
        artifactsBase, `${circuitName}Verifier.sol`, `${contractName}.json`
      );

      if (!fs.existsSync(artifactPath)) {
        console.warn(`    ⚠ Artifact not found: ${artifactPath}`);
        verifiers[circuitType] = null;
        continue;
      }

      const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
      verifiers[circuitType] = await this._deployContract(ethers, artifact, `${circuitName}Verifier`);
    }

    // Deploy main contract
    const flzkpArtifactPath = path.join(
      artifactsBase, "FLZKPReward.sol", `FLZKPReward_${this.circuitId}.json`
    );
    const flzkpArtifact = JSON.parse(fs.readFileSync(flzkpArtifactPath, "utf8"));
    const flzkpReward = await this._deployFLZKPReward(
      ethers, flzkpArtifact, verifiers, `FLZKPReward_${this.circuitId}`
    );

    return {
      commitVerifier: verifiers.commit || null,
      challengeVerifier: verifiers.challenge,
      counterVerifier: verifiers.counter,
      redeemOneShotVerifier: verifiers.redeemOneShot || null,
      redeemMultiShotVerifier: verifiers.redeemMultiShot,
      flzkpReward,
      verifiers,
    };
  }

  async _deployContract(ethers, artifact, name) {
    const factory = new ethers.ContractFactory(artifact.abi, artifact.bytecode, this.signer);
    const contract = await factory.deploy(this.deployOptions);
    await contract.waitForDeployment();
    const address = await contract.getAddress();
    console.log(`✅ ${name} deployed at: ${address}`);
    return contract;
  }

  async _deployFLZKPReward(ethers, artifact, verifiers, name) {
    console.log("  Deploying FLZKPReward contract...");

    // Build address map for all possible constructor params
    const addrMap = {
      _verifierCommit: verifiers.commit ? await verifiers.commit.getAddress() : ZERO_ADDRESS,
      _verifierChallenge: verifiers.challenge ? await verifiers.challenge.getAddress() : ZERO_ADDRESS,
      _verifierCounter: verifiers.counter ? await verifiers.counter.getAddress() : ZERO_ADDRESS,
      _verifierRedeemOneShot: verifiers.redeemOneShot ? await verifiers.redeemOneShot.getAddress() : ZERO_ADDRESS,
      _verifierRedeemMultiShot: verifiers.redeemMultiShot ? await verifiers.redeemMultiShot.getAddress() : ZERO_ADDRESS,
    };

    // Read constructor ABI to determine required arguments dynamically
    const ctorAbi = artifact.abi.find(x => x.type === "constructor");
    const ctorParams = ctorAbi ? ctorAbi.inputs : [];
    const args = [];

    for (const param of ctorParams) {
      if (addrMap[param.name] !== undefined) {
        args.push(addrMap[param.name]);
      } else if (param.name === "_initialCommitment" || param.name === "initialCommitment") {
        let initialCommitment = 0n;
        if (verifiers.commit && this.participants) {
          const STANDARD_SALT = 12345n;
          const helper = new RewardMatrixHelper(this.rounds, this.participants, 16);
          await helper.init();
          const C0 = helper.getCommitment([], [], STANDARD_SALT);
          initialCommitment = helper.uint8ArrayToBigInt(C0);
          console.log(`  Initial commitment (empty state): ${initialCommitment}`);
        }
        args.push(initialCommitment);
      } else {
        console.warn(`  ⚠ Unknown constructor param: ${param.name}, using ZERO_ADDRESS`);
        args.push(ZERO_ADDRESS);
      }
    }

    console.log(`  Constructor args (${args.length}): ${ctorParams.map(p => p.name).join(", ")}`);

    const factory = new ethers.ContractFactory(artifact.abi, artifact.bytecode, this.signer);
    const contract = await factory.deploy(...args, this.deployOptions);
    await contract.waitForDeployment();
    const address = await contract.getAddress();
    console.log(`✅ ${name} deployed at: ${address}`);
    return contract;
  }
}

async function deployFLZKPContracts(ethers, options = {}) {
  const { skipRedeemOneShot = false, ...serviceOptions } = options;
  const service = new ContractDeploymentService(serviceOptions);
  return service.deployContracts(ethers, { skipRedeemOneShot });
}

module.exports = { ContractDeploymentService, deployFLZKPContracts, ZERO_ADDRESS };

const path = require("path");
const snarkjs = require("snarkjs");
const { RewardMatrixHelper } = require("../utils/RewardMatrixHelper");

class ChallengeProver {
  constructor(T, n_participants, chunk_size = 16, wasmPath = undefined, zkeyPath = undefined) {
    this.T = T;
    this.n_participants = n_participants;
    this.chunk_size = chunk_size;
    this.wasmPath = wasmPath ?? path.join(__dirname, "circuits_generator", "Challenge.wasm");
    this.zkeyPath = zkeyPath ?? path.join(__dirname, "circuits_generator", "Challenge_final.zkey");
  }

  async getProof(participants, round_number, matrix, sig_C, agg_pubkey, salt) {
    if (!snarkjs) {
      throw new Error("SnarkJS not found");
    }

    // Validate and pad participants array
    const n_participants = this.n_participants;
    const T = this.T;

    if (participants.length > n_participants) {
      throw new Error(
        `Invalid participants length, required no more than ${n_participants} elements. (current: ${participants.length})`
      );
    }
    const P = new Array(n_participants).fill(0n).map((_, i) => (i >= participants.length ? 0n : participants[i]));

    // Validate and transform matrix to expected format
    if (matrix.length > T) {
      throw new Error(`Invalid matrix length, required no more than ${T} elements. (current: ${matrix.length})`);
    }
    const V = RewardMatrixHelper.transformMatrix(matrix, n_participants, T);

    // Prepare circuit input
    const circuitInput = {
      P,
      round_number,
      V,
      sig_C,
      agg_pubkey,
      salt,
    };

    // console.log("challenge circuitInput: ", JSON.stringify(circuitInput));
    if (process.env.SAVE_INPUT === "true") {
      const dir = `benchmark/input/${this.n_participants}_${this.T}_${this.chunk_size}`;
      require("fs").mkdirSync(dir, { recursive: true });
      require("fs").writeFileSync(`${dir}/challenge_input.json`, JSON.stringify(circuitInput));
    }

    try {
      // Generate ZK proof
      const { proof, publicSignals } = await snarkjs.groth16.fullProve(circuitInput, this.wasmPath, this.zkeyPath);

      // Export proof for Solidity contract
      const ep = await snarkjs.groth16.exportSolidityCallData(proof, publicSignals);
      const eep = JSON.parse("[" + ep + "]");

      return {
        input: circuitInput,
        publicSignals,
        C: eep[3][0],
        aggregator: eep[3][1],
        round_number: eep[3][2],
        a: eep[0],
        b: eep[1],
        c: eep[2],
      };
    } catch (error) {
      if (process.env.DEBUG === "true") {
        console.error("failed to generate proof, input:", JSON.stringify(circuitInput));
      }
      throw error;
    }
  }
}

module.exports = { ChallengeProver };

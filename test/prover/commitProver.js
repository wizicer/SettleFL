const path = require("path");
const snarkjs = require("snarkjs");
const { RewardMatrixHelper } = require("../utils/RewardMatrixHelper");

class CommitProver {
  constructor(T, n_participants, chunk_size = 16, wasmPath = undefined, zkeyPath = undefined) {
    this.T = T;
    this.n_participants = n_participants;
    this.chunk_size = chunk_size;
    this.wasmPath = wasmPath ?? path.join(__dirname, "circuits_generator", "Commit.wasm");
    this.zkeyPath = zkeyPath ?? path.join(__dirname, "circuits_generator", "Commit_final.zkey");
  }

  async getProof(round_number, V2, sig_C1, sig_C2, agg_pubkey, salt, P1_number, P2) {
    if (!snarkjs) {
      throw new Error("SnarkJS not found");
    }

    const n_participants = this.n_participants;
    const T = this.T;

    // Validate and transform V2 matrix to expected format
    if (V2.length > T) {
      throw new Error(`Invalid V2 matrix length, required no more than ${T} elements. (current: ${V2.length})`);
    }
    const transformedV2 = RewardMatrixHelper.transformMatrix(V2, n_participants, T);

    // Validate signature arrays
    if (sig_C1.length !== 3) {
      throw new Error("sig_C1 must have exactly 3 elements [R8x, R8y, S]");
    }
    if (sig_C2.length !== 3) {
      throw new Error("sig_C2 must have exactly 3 elements [R8x, R8y, S]");
    }
    if (agg_pubkey.length !== 2) {
      throw new Error("agg_pubkey must have exactly 2 elements [Ax, Ay]");
    }

    // Validate and pad participant list
    if (P2.length > n_participants) {
      throw new Error(`Invalid P2 length, required no more than ${n_participants} elements. (current: ${P2.length})`);
    }
    const paddedP2 = new Array(n_participants).fill(0n).map((_, i) => (i >= P2.length ? 0n : P2[i]));

    // Prepare circuit input
    const circuitInput = {
      round_number,
      V2: transformedV2,
      sig_C1,
      sig_C2,
      agg_pubkey,
      salt,
      P1_number,
      P2: paddedP2,
    };

    if (process.env.SAVE_INPUT === "true") {
      const dir = `benchmark/input/${this.n_participants}_${this.T}_${this.chunk_size}`;
      require("fs").mkdirSync(dir, { recursive: true });
      require("fs").writeFileSync(`${dir}/commit_input.json`, JSON.stringify(circuitInput));
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
        C1: eep[3][0],
        C2: eep[3][1],
        aggregator: eep[3][2],
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

module.exports = { CommitProver };

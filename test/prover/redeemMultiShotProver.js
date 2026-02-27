const path = require("path");
const snarkjs = require("snarkjs");
const { RewardMatrixHelper } = require("../utils/RewardMatrixHelper");

class RedeemMultiShotProver {
  constructor(n_participants, T, chunk_size = 16, batch_size, wasmPath = undefined, zkeyPath = undefined) {
    this.n_participants = n_participants;
    this.T = T;
    this.chunk_size = chunk_size;
    this.batch_size = batch_size;
    this.wasmPath = wasmPath ?? path.join(__dirname, "circuits_generator", "RedeemMultiShot.wasm");
    this.zkeyPath = zkeyPath ?? path.join(__dirname, "circuits_generator", "RedeemMultiShot_final.zkey");
  }

  async getProof(round_number, n_shots, V, sig_C, agg_pubkey, salt, P) {
    if (!snarkjs) {
      throw new Error("SnarkJS not found");
    }

    const n_participants = this.n_participants;
    const T = this.T;
    const batch_size = this.batch_size;

    // Validate and transform V matrix to expected format
    if (V.length > T) {
      throw new Error(`Invalid V matrix length, required no more than ${T} elements. (current: ${V.length})`);
    }
    const transformedV = RewardMatrixHelper.transformMatrix(V, n_participants, T);

    // Validate signature array
    if (sig_C.length !== 3) {
      throw new Error("sig_C must have exactly 3 elements [R8x, R8y, S]");
    }
    if (agg_pubkey.length !== 2) {
      throw new Error("agg_pubkey must have exactly 2 elements [Ax, Ay]");
    }

    // Validate participant array
    if (P.length > n_participants) {
      throw new Error(`Invalid P length, required no more than ${n_participants} elements. (current: ${P.length})`);
    }
    const paddedP = new Array(n_participants).fill(0n).map((_, i) => (i >= P.length ? 0n : BigInt(P[i])));

    if (n_shots === undefined) {
      throw new Error("n_shots is undefined");
    }

    // Prepare circuit input
    const circuitInput = {
      round_number,
      n_shots,
      V: transformedV,
      sig_C,
      agg_pubkey,
      salt,
      P: paddedP,
    };

    if (process.env.SAVE_INPUT === "true") {
      const dir = `benchmark/input/${this.n_participants}_${this.T}_${this.chunk_size}`;
      require("fs").mkdirSync(dir, { recursive: true });
      require("fs").writeFileSync(`${dir}/redeemMultiShot_input.json`, JSON.stringify(circuitInput));
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
        batch_S: eep[3].slice(2, 2 + batch_size),
        batch_P: eep[3].slice(2 + batch_size, 2 + batch_size * 2),
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

module.exports = { RedeemMultiShotProver };

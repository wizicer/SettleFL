const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");
const snarkjs = require("snarkjs");
const { RewardMatrixHelper } = require("../utils/RewardMatrixHelper");

class CounterProverRs {
  constructor(T, n_participants, chunk_size = 16, wasmPath = undefined, zkeyPath = undefined) {
    this.T = T;
    this.n_participants = n_participants;
    this.chunk_size = chunk_size;
    this.wasmPath = wasmPath ?? path.join(__dirname, "circuits_generator", "Counter.wasm");
    this.zkeyPath = zkeyPath ?? path.join(__dirname, "circuits_generator", "Counter_final.zkey");

    // Get Rapidsnark binary path from environment
    this.rapidsnarkBin = process.env.RAPID_SNARK_BIN_PATH;
    if (!this.rapidsnarkBin) {
      throw new Error("RAPID_SNARK_BIN_PATH environment variable not set");
    }
  }

  async generateWitness(circuitInput) {
    // Create temporary witness file
    const witnessPath = path.join(__dirname, "temp_witness.wtns");

    // Write circuit input to a temporary JSON file for witness generation
    const inputPath = path.join(__dirname, "temp_input.json");
    fs.writeFileSync(
      inputPath,
      JSON.stringify(circuitInput, (key, value) => (typeof value === "bigint" ? value.toString() : value))
    );

    return new Promise((resolve, reject) => {
      // Use snarkjs to generate witness (this is still needed for Rapidsnark)
      const snarkjs = spawn("npx", ["snarkjs", "wtns", "calculate", this.wasmPath, inputPath, witnessPath]);

      let stderr = "";
      snarkjs.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      snarkjs.on("close", (code) => {
        // Clean up input file
        try {
          fs.unlinkSync(inputPath);
        } catch (e) {}

        if (code === 0) {
          resolve(witnessPath);
        } else {
          reject(new Error(`Witness generation failed: ${stderr}`));
        }
      });
    });
  }

  async rapidsnarkProve(zkeyPath, witnessPath) {
    const proofPath = path.join(__dirname, "temp_proof.json");
    const publicPath = path.join(__dirname, "temp_public.json");

    return new Promise((resolve, reject) => {
      // Rapidsnark command: rapidsnark zkey witness proof public
      const rapidsnark = spawn(this.rapidsnarkBin, [zkeyPath, witnessPath, proofPath, publicPath]);

      let stderr = "";
      rapidsnark.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      rapidsnark.on("close", (code) => {
        if (code === 0) {
          try {
            const proofContent = fs.readFileSync(proofPath, "utf8");
            const publicContent = fs.readFileSync(publicPath, "utf8");

            // Parse proof and public signals (find the end of JSON)
            const proofEndIndex = proofContent.lastIndexOf("}") + 1;
            const publicEndIndex = publicContent.lastIndexOf("]") + 1;

            const proofJson = proofContent.substring(0, proofEndIndex);
            const publicJson = publicContent.substring(0, publicEndIndex);

            const proof = JSON.parse(proofJson);
            const publicSignals = JSON.parse(publicJson);

            // Clean up temporary files
            try {
              fs.unlinkSync(proofPath);
            } catch (e) {}
            try {
              fs.unlinkSync(publicPath);
            } catch (e) {}
            try {
              fs.unlinkSync(witnessPath);
            } catch (e) {}

            resolve({ proof, publicSignals });
          } catch (e) {
            reject(new Error(`Failed to parse proof files: ${e.message}`));
          }
        } else {
          // Clean up witness file
          try {
            fs.unlinkSync(witnessPath);
          } catch (e) {}
          reject(new Error(`Rapidsnark failed: ${stderr}`));
        }
      });
    });
  }

  async getProof(round_number, V2, participants2, P1_number, sig_C1, sig_C2, agg_pubkey, salt) {
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

    // Validate and pad participant
    if (participants2.length > n_participants) {
      throw new Error(
        `Invalid participant length, required no more than ${n_participants} elements. (current: ${participants2.length})`
      );
    }
    const P2 = new Array(n_participants).fill(0n).map((_, i) => (i >= participants2.length ? 0n : participants2[i]));

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

    // Prepare circuit input
    const circuitInput = {
      round_number,
      V2: transformedV2,
      P1_number,
      P2,
      sig_C1,
      sig_C2,
      agg_pubkey,
      salt,
    };

    if (process.env.SAVE_INPUT === "true") {
      const dir = `benchmark/input/${this.n_participants}_${this.T}_${this.chunk_size}`;
      require("fs").mkdirSync(dir, { recursive: true });
      require("fs").writeFileSync(`${dir}/counter_input.json`, JSON.stringify(circuitInput));
    }

    try {
      // Generate witness
      const witnessPath = await this.generateWitness(circuitInput);

      // Generate proof using Rapidsnark
      const { proof, publicSignals } = await this.rapidsnarkProve(this.zkeyPath, witnessPath);

      // Use SnarkJS to format proof for Solidity contract (ensures compatibility)
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

module.exports = { CounterProverRs };

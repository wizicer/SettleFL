const { buildPoseidon, buildEddsa } = require("circomlibjs");

class RewardMatrixHelper {
  constructor(T, n_participants, chunk_size = 16) {
    this.T = T;
    this.n_participants = n_participants;
    this.chunk_size = chunk_size;
    this.poseidon = null;
    this.eddsa = null;
  }

  async init() {
    this.poseidon = await buildPoseidon();
    this.eddsa = await buildEddsa();
  }

  hashV(matrix, salt) {
    const V = this.transformMatrix(matrix);

    if (this.poseidon === null) {
      throw new Error("Poseidon is not initialized");
    }

    const chunks = this.buildChunkMap(this.T, this.n_participants, this.chunk_size);
    const out = new Array(chunks.length).fill(0n);

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const values = chunk.map((c) => this.convertChunkCellToValue(c, out, V, salt));
      const hash = this.poseidon(values);
      out[i] = this.uint8ArrayToBigInt(hash);
    }

    return out[out.length - 1];
  }

  /**
   * Compute a Poseidon hash over an array of field elements using a modular strategy.
   * Splits input into chunks of 16 elements (or less for the last chunk),
   * hashes each chunk with Poseidon, then reduces the chunk hashes using binary tree hashing.
   *
   * @param {bigint[]} input - Array of field elements (as BigInt)
   * @param {object} poseidon - Poseidon hash object with a function `poseidon(inputs: bigint[]): bigint`
   * @returns {bigint} - Final Poseidon hash of the input
   */
  poseidonModular(input) {
    const CHUNK_SIZE = 16;
    const numElements = input.length;

    if (numElements === 0) {
      throw new Error("Input array must not be empty");
    }

    // Split into chunks and hash each chunk
    const chunkHashes = [];
    for (let i = 0; i < numElements; i += CHUNK_SIZE) {
      const chunk = input.slice(i, i + CHUNK_SIZE);
      const chunkHash = this.poseidon(chunk);
      chunkHashes.push(chunkHash);
    }

    // Reduce with binary tree-style Poseidon hashing
    let result = chunkHashes[0];
    for (let i = 1; i < chunkHashes.length; i++) {
      result = this.poseidon([result, chunkHashes[i]]);
    }

    return result;
  }

  getCommitment(matrix, participants, salt) {
    const hashV = this.hashV(matrix, 0);
    const ps = new Array(this.n_participants).fill(0n).map((_, i) => (i < participants.length ? BigInt(participants[i]) : 0n));
    const hashP = this.poseidonModular(ps);
    return this.poseidon([this.bigintToUint8Array(hashV), hashP, this.bigintToUint8Array(salt)]);
  }

  emptyMatrix() {
    return [];
  }

  appendRow(matrix, row) {
    return matrix.concat([row]);
  }

  calculateRewards(matrix) {
    const rewards = Array(matrix[0].length).fill(0n);
    for (let r = 0; r < matrix.length; r++) {
      const row = matrix[r];
      for (let p = 0; p < row.length; p++) {
        const val = row[p];
        if (val > 0n) {
          rewards[p] += val;
        }
      }
    }
    return rewards;
  }

  hashPubkey(pubkey) {
    if (this.poseidon === null) {
      throw new Error("Poseidon is not initialized");
    }
    return this.uint8ArrayToBigInt(this.poseidon([pubkey.Ax, pubkey.Ay]));
  }

  getPubkey(privateKey) {
    if (this.eddsa === null || this.poseidon === null) {
      throw new Error("Eddsa or Poseidon is not initialized");
    }

    const pub = this.eddsa.prv2pub(typeof privateKey === "bigint" ? this.bigintToUint8Array(privateKey) : privateKey);
    return {
      Ax: this.uint8ArrayToBigInt(pub[0]),
      Ay: this.uint8ArrayToBigInt(pub[1]),
    };
  }

  sign(privateKey, msg) {
    if (this.eddsa === null || this.poseidon === null) {
      throw new Error("Eddsa or Poseidon is not initialized");
    }

    const sig = this.eddsa.signPoseidon(
      typeof privateKey === "bigint" ? this.bigintToUint8Array(privateKey) : privateKey,
      this.poseidon.F.e(msg)
    );

    return {
      R8x: this.uint8ArrayToBigInt(sig.R8[0]),
      R8y: this.uint8ArrayToBigInt(sig.R8[1]),
      S: sig.S,
    };
  }

  verify(pubkey, msg, signature) {
    if (this.eddsa === null) {
      throw new Error("Eddsa is not initialized");
    }

    return this.eddsa.verifyPoseidon(
      this.poseidon.F.e(msg),
      {
        R8: [signature.R8x, signature.R8y],
        S: signature.S,
      },
      [pubkey.Ax, pubkey.Ay]
    );
  }

  buildChunkMap(T, n_participants, chunk_size) {
    const chunk_data_size = chunk_size - 1;
    const total_inputs = T * n_participants;
    const chunk_num = Math.ceil((total_inputs - chunk_data_size) / chunk_data_size) + 1;

    const chunks = Array.from({ length: chunk_num }, () => new Array(chunk_size).fill("-"));

    for (let i = 0; i < chunk_num; i++) {
      chunks[i][0] = i === 0 ? { type: "seed" } : { type: "out", i: i - 1 };
      for (let j = 1; j < chunk_size; j++) {
        const input_index = i * chunk_data_size + (j - 1);
        if (input_index >= total_inputs) {
          chunks[i][j] = { type: "0" };
        } else {
          const m = Math.floor(input_index / T);
          const n = input_index % T;
          chunks[i][j] = { type: "V", m, n };
        }
      }
    }
    return chunks;
  }

  convertChunkCellToValue(chunkCell, out, V, salt) {
    if (chunkCell.type === "0") {
      return 0n;
    } else if (chunkCell.type === "seed") {
      return salt;
    } else if (chunkCell.type === "out") {
      return out[chunkCell.i];
    } else if (chunkCell.type === "V") {
      return V[chunkCell.m]?.[chunkCell.n] || 0n;
    } else {
      return 0n;
    }
  }

  uint8ArrayToBigInt(arr) {
    if (this.poseidon === null) {
      throw new Error("Poseidon is not initialized");
    }
    return this.poseidon.F.toObject(arr);
  }

  bigintToUint8Array(num) {
    if (this.poseidon === null) {
      throw new Error("Poseidon is not initialized");
    }
    return this.poseidon.F.fromObject(num);
  }

  transformMatrix(matrix) {
    return RewardMatrixHelper.transformMatrix(matrix, this.n_participants, this.T);
  }

  static transformMatrix(matrix, n_participants, T) {
    return new Array(n_participants)
      .fill(null)
      .map(() => new Array(T).fill(0n))
      .map((row, i) =>
        row.map((_, j) => {
          if (j < matrix.length && i < matrix[j].length) {
            return BigInt(matrix[j][i]);
          }
          return 0n;
        })
      );
  }

  createTestMatrix(round, participants) {
    const V = Array(round)
      .fill(0)
      .map(() => Array(participants).fill(0n));

    for (let i = 0; i < round; i++) {
      for (let j = 0; j < participants; j++) {
        V[i][j] = BigInt(i * participants + j + 1);
      }
    }

    return V;
  }

  generateTestParticipants(count) {
    const participants = [];
    for (let i = 0; i < count; i++) {
      const addr = `0x${(i + 1).toString(16).padStart(40, "0")}`;
      participants.push(addr);
    }
    return participants;
  }
}

module.exports = { RewardMatrixHelper };

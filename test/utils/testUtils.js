const { buildPoseidon, buildEddsa } = require("circomlibjs");
const { ethers } = require("ethers");

class TestUtils {
  constructor() {
    this.poseidon = null;
    this.eddsa = null;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;
    this.poseidon = await buildPoseidon();
    this.eddsa = await buildEddsa();
    this.initialized = true;
  }

  generatePrivateKey() {
    return new Uint8Array(32).map(() => Math.floor(Math.random() * 256));
  }

  generatePublicKey(privateKey) {
    return this.eddsa.prv2pub(privateKey);
  }

  signMessage(privateKey, message) {
    const msgBigInt = typeof message === "string" ? BigInt(message) : message;
    return this.eddsa.signPoseidon(privateKey, msgBigInt);
  }

  createMockProof() {
    return {
      a: [
        "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
      ],
      b: [
        [
          "0x1111111111111111111111111111111111111111111111111111111111111111",
          "0x2222222222222222222222222222222222222222222222222222222222222222",
        ],
        [
          "0x3333333333333333333333333333333333333333333333333333333333333333",
          "0x4444444444444444444444444444444444444444444444444444444444444444",
        ],
      ],
      c: [
        "0x5555555555555555555555555555555555555555555555555555555555555555",
        "0x6666666666666666666666666666666666666666666666666666666666666666",
      ],
    };
  }

  generateParticipants(count = 3) {
    const participants = [];
    for (let i = 0; i < count; i++) {
      participants.push(ethers.Wallet.createRandom().address);
    }
    return participants;
  }

  generateRewards(participantCount, totalReward = ethers.parseEther("1.0")) {
    const rewards = [];
    const baseReward = totalReward / BigInt(participantCount);
    let sum = 0n;

    for (let i = 0; i < participantCount - 1; i++) {
      rewards.push(baseReward);
      sum += baseReward;
    }

    rewards.push(totalReward - sum);
    return rewards;
  }

  generateChallengePublicSignals(participants, rewards, round, commitment, pubHash) {
    const MAX_PARTICIPANTS = 10;
    const signals = new Array(23).fill(0);

    for (let i = 0; i < participants.length && i < MAX_PARTICIPANTS; i++) {
      signals[i] = BigInt(participants[i]);
    }

    signals[MAX_PARTICIPANTS] = BigInt(round);
    signals[MAX_PARTICIPANTS + 1] = BigInt(commitment);

    for (let i = 0; i < rewards.length && i < MAX_PARTICIPANTS; i++) {
      signals[i + 2 + MAX_PARTICIPANTS] = BigInt(rewards[i]);
    }

    signals[MAX_PARTICIPANTS * 2 + 2] = BigInt(pubHash);

    return signals;
  }

  generateCounterPublicSignals(round, commitment1, commitment2, pubHash) {
    return [BigInt(round), BigInt(commitment1), BigInt(commitment2), BigInt(pubHash)];
  }

  generateCommitment(data) {
    const inputs = Array.isArray(data) ? data : [data];
    const hash = this.poseidon(inputs);
    return this.poseidon.F.toString(hash);
  }

  generateTestMatrix(rows, cols) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) {
        row.push(Math.floor(Math.random() * 1000));
      }
      matrix.push(row);
    }
    return matrix;
  }

  bigIntToHex(value) {
    return "0x" + value.toString(16).padStart(64, "0");
  }

  generateSalt() {
    return Math.floor(Math.random() * 1000000);
  }

  generatePubHash() {
    return this.generateCommitment([Math.floor(Math.random() * 1000000)]);
  }

  async advanceTime(seconds) {
    await ethers.provider.send("evm_increaseTime", [seconds]);
    await ethers.provider.send("evm_mine", []);
  }

  async getCurrentTimestamp() {
    const block = await ethers.provider.getBlock("latest");
    return block.timestamp;
  }

  formatProof(proof) {
    return {
      a: proof.a,
      b: proof.b,
      c: proof.c,
    };
  }

  toAddress(hex32) {
    if (typeof hex32 !== "string" || !/^0x[0-9a-fA-F]{64}$/.test(hex32)) {
      throw new Error("Input must be a 32-byte hex string, source: " + hex32);
    }
    const address = "0x" + hex32.slice(-40);
    return address.toLowerCase();
  }

  async generateFlowTestData(participantCount = 3, rounds = 2) {
    const participants = this.generateParticipants(participantCount);
    const totalReward = ethers.parseEther("1.0");
    const rewards = this.generateRewards(participantCount, totalReward);
    const salt = this.generateSalt();
    const pubHash = this.generatePubHash();

    const commitments = [];
    for (let i = 0; i < rounds; i++) {
      const commitment = this.generateCommitment([salt, i, ...participants]);
      commitments.push(commitment);
    }

    return {
      participants,
      rewards,
      totalReward,
      salt,
      pubHash,
      commitments,
      mockProof: this.createMockProof(),
    };
  }
}

module.exports = { TestUtils };

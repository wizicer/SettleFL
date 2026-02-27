# SettleFL: Artifact Evaluation & Disclosure Build

> **Note on Source Code Availability:**
> This repository contains a **pre-compiled disclosure build** of the SettleFL protocol. To protect core intellectual property during the review process, we are currently providing only the compiled artifacts. This build serves to demonstrate that the end-to-end system described in our paper is fully functional and verifiable in a real-world environment.
> **The complete source code (including all `.circom` circuits and `.sol` smart contracts) will be fully open-sourced upon the paper's formal acceptance.**

## About SettleFL

This repository demonstrates the **SettleFL** protocol family, designed to resolve the conflict between costly on-chain execution and high-frequency Federated Learning (FL) updates. It navigates the scalability-finality trade-off through two interchangeable variants:

* **SettleFL-CC (Commit-and-Challenge):** An optimistic execution variant that minimizes on-chain costs. It relies on a dispute-driven arbitration mechanism where participants can challenge the aggregator's commitment with Zero-Knowledge (ZK) proofs if misbehavior is detected.

* **SettleFL-CP (Commit-with-Proof):** A validity-proof variant that guarantees instant finality. In this mode, the aggregator attaches a SNARK proof to every per-round commitment, ensuring upfront correctness without requiring a dispute window.



## What's Included vs. Not Included

**Included in this Artifact Build:**

* **Pre-compiled ZK circuit artifacts:** WASM witness generators and `zkey` proving keys configured for a `10_10_16_10` parameter set.
* **Pre-compiled smart contract artifacts:** ABI and bytecode to deploy the verifier and protocol logic.
* **JavaScript test harness:** Includes provers, the federated learning training simulation, and the end-to-end test runner.

**Excluded (To be released upon acceptance):**

* Circom circuit source code (`.circom` files).
* Solidity smart contract source code (`.sol` files).
* Comprehensive benchmark source code

## Circuit Parameters

The pre-compiled circuits in this build are constrained to the following parameters:

| Parameter | Value |
| --- | --- |
| **Max Participants** | 10 |
| **Max Rounds** | 10 |
| **Chunk Size** | 16 |
| **Payout Batch Size** | 10 |

## Quick Start

**Prerequisites:** Node.js >= 20

**1. Install dependencies**

```bash
npm install
```

**2. Configure Environment Variables**
Copy the example environment file to create your local `.env` configuration:

```bash
cp .env.example .env
```

*(By default, this file contains well-known test accounts for the local Hardhat network. If you intend to run tests on the Sepolia public testnet, please refer to the "Network Selection" section below).*

**3. Download the Dataset**
Before running the training simulation, you must download the MNIST dataset, which serves as the main benchmark for our end-to-end evaluation:

```bash
npm run download:mnist
```

**4. Start a local Hardhat blockchain** *(Run this in a separate terminal)*

```bash
npm run chain
```

**5. Run the SettleFL-CC (Commit-and-Challenge) Test**

```bash
npm run test:challenge-counter
```

*This executes a 5-participant, 3-round FL job on the MNIST dataset using the optimistic protocol. It simulates local training, aggregator commitments, a malicious challenge, an honest counter-proof generation, and final ZK-verified reward distribution.*

**6. Run the SettleFL-CP (Commit-with-Proof) Test**

```bash
npm run test:commit-proof
```

*This executes the validity-proof protocol. Every commit round will automatically generate and verify a ZK proof on-chain, proving state transition validity instantly without a challenge phase.*

## Customizing the Test Run

You can override the default test simulation parameters (ensure you do not exceed the max circuit limits of 10 participants/10 rounds):

```bash
npm run test:real -- --participants 3 --rounds 2 --max-participants 10 --max-rounds 10 --payout-batch-size 10
```

## Network Selection (Local vs. Sepolia)

By default, the testing harness executes on a local Hardhat node using the pre-funded accounts specified in your `.env` file. We also fully support executing these workflows on the public **Ethereum Sepolia testnet**.

To evaluate the protocol on Sepolia:

1. Open your `.env` file.
2. Replace the values for `OWNER_PRIVATE_KEY_sepolia` and `CHALLENGER_PRIVATE_KEY_sepolia` with your own actual private keys. **Ensure these accounts are funded with Sepolia test ETH.**
3. Append the `--network sepolia` flag to your test commands. For example:
```bash
npm run test:challenge-counter -- --network sepolia

```

## FAQ

**Q: My test fails with a generic error like `❌ Test failed: Error: X training tasks failed` during local training. What should I do?**

**A:**
When verbose logging is disabled, the system might mask the actual underlying error. If you are running this on a new machine or a recently updated environment, this is highly likely caused by an incompatible Node.js version.

**How to confirm:**
Enable verbose mode or check your detailed logs via `--log-level debug`. If you spot the following error:
`TypeError: (0 , util_1.isNullOrUndefined) is not a function`

**The Core Reason:**
Your Node.js version is too new (v23 or higher). The underlying `@tensorflow/tfjs-node` library relies on a legacy Node.js built-in function called `util.isNullOrUndefined`. This function was completely removed starting from Node.js v23, causing the training tasks to crash silently.

**The Solution:**
You need to downgrade your Node.js to a stable LTS (Long Term Support) version. We highly recommend **Node.js v22**.

## Citation

If you find this project or our paper interesting, please consider citing our extended version on arXiv:

```bibtex
@misc{liang2026settlefltrustlessscalablereward,
      title={SettleFL: Trustless and Scalable Reward Settlement Protocol for Federated Learning on Permissionless Blockchains (Extended version)}, 
      author={Shuang Liang and Yang Hua and Linshan Jiang and Peishen Yan and Tao Song and Bin Yao and Haibing Guan},
      year={2026},
      eprint={2602.23167},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.23167}, 
}

```
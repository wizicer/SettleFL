const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

class CircuitCompiler {
  constructor(circuitConfig) {
    this.circuitConfig = circuitConfig || {};
    this.circuitDir = this.circuitConfig.circuitDir || path.join(__dirname, "..", "out");
    if (!fs.existsSync(this.circuitDir)) {
      fs.mkdirSync(this.circuitDir, { recursive: true });
    }
  }

  getCircuitFileContent(parameters) {
    const { templateName, includePath, publicParameters } = this.circuitConfig;

    let publicOutputsStr = "";
    if (publicParameters && publicParameters.length > 0) {
      publicOutputsStr = `public [${publicParameters.join(", ")}]`;
    }

    // Build parameters string
    const paramStr = parameters.join(", ");

    // Handle empty public parameters - don't include braces if no public outputs
    const componentDeclaration = publicOutputsStr ? `component main { ${publicOutputsStr} }` : `component main`;

    // Use relative path from circuit directory to include path
    const relativeIncludePath = path.relative(this.circuitDir, includePath).replace(/\\/g, "/");

    return `pragma circom 2.2.0;
include "${relativeIncludePath}";

${componentDeclaration} = ${templateName}(${paramStr});`;
  }

  stripAnsi(str) {
    return str.replace(/\x1b\[[0-9;]*m/g, "");
  }

  parseCircomOutput(output) {
    const lines = output.split("\n");
    const data = {};

    for (const line of lines) {
      if (this.stripAnsi(line).includes("template instances:")) {
        data.template_instances = parseInt(line.split(":")[1].trim());
      } else if (line.includes("non-linear constraints:")) {
        data.non_linear_constraints = parseInt(line.split(":")[1].trim());
      } else if (line.includes("linear constraints:")) {
        data.linear_constraints = parseInt(line.split(":")[1].trim());
      } else if (line.includes("public inputs:")) {
        data.public_inputs = parseInt(line.split(":")[1].trim());
      } else if (line.includes("private inputs:")) {
        data.private_inputs = parseInt(line.split(":")[1].trim().split(" ")[0]);
      } else if (line.includes("public outputs:")) {
        data.public_outputs = parseInt(line.split(":")[1].trim());
      } else if (line.includes("wires:")) {
        data.wires = parseInt(line.split(":")[1].trim());
      } else if (line.includes("labels:")) {
        data.labels = parseInt(line.split(":")[1].trim());
      }
    }

    return data;
  }

  async compileCircuit(parameters) {
    // Generate circuit file
    const fileContent = this.getCircuitFileContent(parameters);
    const circuitFilePath = path.join(this.circuitDir, "circuit.circom");
    fs.writeFileSync(circuitFilePath, fileContent, "utf8");
    let parsedData;

    try {
      // Compile circuit
      const participants = Number(parameters?.[0]);
      const rounds = Number(parameters?.[1]);
      const tempValue = 88 * rounds * participants;
      const shouldUseWasm = !(Number.isFinite(tempValue) && tempValue > 10_000_000);
      const wasmFlag = shouldUseWasm ? " --wasm" : "";

      const output = execSync(`circom circuit.circom --r1cs -c${wasmFlag} -l ${this.circuitConfig.libPath ?? "./node_modules"}`, {
        stdio: "pipe",
        cwd: this.circuitDir,
      }).toString();

      parsedData = this.parseCircomOutput(output);

      // Setup zKey (if ptau file exists)
      const ptauPath = path.join(__dirname, "..", "..", "pot_final.ptau");
      if (fs.existsSync(ptauPath)) {
        execSync(`snarkjs groth16 setup circuit.r1cs ${ptauPath} circuit_final.zkey`, {
          stdio: "pipe",
          cwd: this.circuitDir,
        });

        // Export verification key
        execSync("snarkjs zkey export verificationkey circuit_final.zkey verification_key.json", {
          stdio: "pipe",
          cwd: this.circuitDir,
        });

        // Export Solidity verifier
        execSync("snarkjs zkey export solidityverifier circuit_final.zkey verifier.sol", {
          stdio: "pipe",
          cwd: this.circuitDir,
        });
      }

      return {
        success: true,
        circuitDir: this.circuitDir,
        wasmPath: path.join(this.circuitDir, "circuit_js", "circuit.wasm"),
        zkeyPath: path.join(this.circuitDir, "circuit_final.zkey"),
        verifierPath: path.join(this.circuitDir, "verifier.sol"),
        constraints: parsedData.non_linear_constraints,
        wires: parsedData.wires,
        templateInstances: parsedData.template_instances,
        publicInputs: parsedData.public_inputs,
        privateInputs: parsedData.private_inputs,
        publicOutputs: parsedData.public_outputs,
      };
    } catch (error) {
      if (parsedData) {
        this.saveCircuitStats(parsedData);
      }
      console.error("Error compiling circuit: ", error);
      return {
        success: false,
        error: error.message,
        circuitDir: this.circuitDir,
      };
    }
  }

  async generateWitness(input) {
    // Use the output paths and names as in the rest of this file
    const inputPath = path.join(this.circuitDir, "input.json");
    fs.writeFileSync(inputPath, JSON.stringify(input, null, 2));

    const witnessPath = path.join(this.circuitDir, "witness.wtns");
    const wasmPath = path.join(this.circuitDir, "circuit_js", "circuit.wasm");
    const generateWitnessJs = path.join(this.circuitDir, "circuit_js", "generate_witness.js");
    const generateCmd = `node ${generateWitnessJs} ${wasmPath} ${inputPath} ${witnessPath}`;

    try {
      execSync(generateCmd, { stdio: "pipe" });
    } catch (error) {
      if (fs.existsSync(inputPath)) {
        fs.unlinkSync(inputPath);
      }
      throw new Error(`Failed to generate witness: ${error.message}`);
    }

    fs.unlinkSync(inputPath);

    return witnessPath;
  }

  async parseWitnessFile(witnessPath) {
    const witnessJsonPath = path.join(this.circuitDir, "witness.json");
    const exportCmd = `snarkjs wtns export json ${witnessPath} ${witnessJsonPath}`;

    try {
      execSync(exportCmd, { stdio: "pipe" });
    } catch (error) {
      throw new Error(`Failed to export witness to JSON: ${error.message}`);
    }

    const witnessData = JSON.parse(fs.readFileSync(witnessJsonPath, "utf8"));

    fs.unlinkSync(witnessJsonPath);

    return witnessData;
  }

  getCircuitStats() {
    const statsPath = path.join(this.circuitDir, "circuit_stats.json");
    if (fs.existsSync(statsPath)) {
      return JSON.parse(fs.readFileSync(statsPath, "utf8"));
    }

    return null;
  }

  saveCircuitStats(stats) {
    const statsPath = path.join(this.circuitDir, "circuit_stats.json");

    fs.writeFileSync(statsPath, JSON.stringify(stats, null, 2));
  }
}

module.exports = { CircuitCompiler };

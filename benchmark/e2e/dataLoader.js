const fs = require("fs");
const path = require("path");

class DataLoader {
  constructor(dataDir, logFn, maxDataSizePerParticipant = 200) {
    this.dataDir = dataDir;
    this.log = logFn;
    this.maxDataSizePerParticipant = maxDataSizePerParticipant;
  }

  async load(dataset, participantCount) {
    this.log("info", `Loading ${dataset.toUpperCase()} data...`);
    let trainData, testData;

    switch (dataset) {
      case "mnist":
        ({ trainData, testData } = await this.loadMnistData());
        break;
      case "cifar10":
        ({ trainData, testData } = await this.loadCifar10Data());
        break;
      case "fashion-mnist":
        ({ trainData, testData } = await this.loadFashionMnistData());
        break;
      case "bank-marketing":
        ({ trainData, testData } = await this.loadBankMarketingData());
        break;
      case "cora":
        // Graph dataset: returns graph object instead of sample arrays
        return await this.loadCoraData(participantCount);
      case "sent140":
        // Text dataset: returns sequence-indexed samples
        return await this.loadSent140Data(participantCount);
      default:
        throw new Error(`Unsupported dataset: ${dataset}`);
    }

    const participantData = this.splitDataForParticipants(trainData, participantCount);
    this.log(
      "info",
      `${dataset.toUpperCase()} data loaded. Training samples: ${trainData.length}, Test samples: ${testData.length}`
    );

    return { participantData, testData };
  }

  async loadMnistData() {
    const trainDataPath = path.join(this.dataDir, "mnist_train.csv");
    const testDataPath = path.join(this.dataDir, "mnist_test.csv");

    if (!fs.existsSync(trainDataPath) || !fs.existsSync(testDataPath)) {
      throw new Error("MNIST data files not found. Please ensure mnist_train.csv and mnist_test.csv are in the data directory.");
    }

    const trainData = await this.loadCSVData(trainDataPath);
    const testData = await this.loadCSVData(testDataPath);
    return { trainData, testData };
  }

  async loadCifar10Data() {
    const dataPath = path.join(this.dataDir, "cifar-10-batches-bin");
    if (!fs.existsSync(dataPath)) {
      throw new Error("CIFAR-10 data not found. Please run `npm run download:cifar10` first.");
    }

    const trainData = [];
    for (let i = 1; i <= 5; i++) {
      const filePath = path.join(dataPath, `data_batch_${i}.bin`);
      trainData.push(...this.loadCifar10Batch(filePath));
    }

    const testData = this.loadCifar10Batch(path.join(dataPath, "test_batch.bin"));
    return { trainData, testData };
  }

  async loadFashionMnistData() {
    const dataPath = path.join(this.dataDir, "fashion-mnist");
    if (!fs.existsSync(dataPath)) {
      throw new Error("Fashion-MNIST data not found. Please run `npm run download:fashion-mnist` first.");
    }

    const trainImages = this.loadIdxFile(path.join(dataPath, "train-images-idx3-ubyte"));
    const trainLabels = this.loadIdxFile(path.join(dataPath, "train-labels-idx1-ubyte"));
    const testImages = this.loadIdxFile(path.join(dataPath, "t10k-images-idx3-ubyte"));
    const testLabels = this.loadIdxFile(path.join(dataPath, "t10k-labels-idx1-ubyte"));

    const trainData = trainLabels.map((label, i) => ({ label, pixels: trainImages[i] }));
    const testData = testLabels.map((label, i) => ({ label, pixels: testImages[i] }));
    return { trainData, testData };
  }

  loadCifar10Batch(filePath) {
    const buffer = fs.readFileSync(filePath);
    const data = [];
    const recordSize = 3073;
    for (let i = 0; i < buffer.length; i += recordSize) {
      const label = buffer[i];
      const pixels = Array.from(buffer.slice(i + 1, i + 1 + 3072));
      data.push({ label, pixels });
    }
    return data;
  }

  loadIdxFile(filePath) {
    const buffer = fs.readFileSync(filePath);
    const magicNumber = buffer.readInt32BE(0);
    const numItems = buffer.readInt32BE(4);
    const numDimensions = magicNumber & 0xff;

    if (numDimensions === 1) {
      return Array.from(buffer.slice(8));
    } else if (numDimensions === 3) {
      const rows = buffer.readInt32BE(8);
      const cols = buffer.readInt32BE(12);
      const imageSize = rows * cols;
      const images = [];
      let offset = 16;
      for (let i = 0; i < numItems; i++) {
        images.push(Array.from(buffer.slice(offset, offset + imageSize)));
        offset += imageSize;
      }
      return images;
    }
    throw new Error(`Unsupported IDX file format with magic number ${magicNumber}`);
  }

  async loadCSVData(filePath) {
    this.log("debug", `Loading CSV data from: ${filePath}`);
    const fileContent = fs.readFileSync(filePath, "utf8");
    const lines = fileContent.split("\n").filter((line) => line.trim());
    const dataLines = lines[0].includes("label") ? lines.slice(1) : lines;

    const data = [];
    for (const line of dataLines) {
      const values = line.split(",");
      const label = parseInt(values[0], 10);
      const pixels = values.slice(1).map((v) => parseInt(v, 10));
      data.push({ label, pixels });
    }
    return data;
  }

  async loadBankMarketingData() {
    const dataPath = path.join(this.dataDir, "bank-marketing");
    const trainXPath = path.join(dataPath, "train_X.csv");
    const trainYPath = path.join(dataPath, "train_y.csv");
    const testXPath = path.join(dataPath, "test_X.csv");
    const testYPath = path.join(dataPath, "test_y.csv");

    for (const p of [trainXPath, trainYPath, testXPath, testYPath]) {
      if (!fs.existsSync(p)) {
        throw new Error(`Bank Marketing data file not found: ${p}\nPlease run \`npm run download:bank-marketing\` first.`);
      }
    }

    const trainData = this.loadTabularCSV(trainXPath, trainYPath);
    const testData = this.loadTabularCSV(testXPath, testYPath);

    this.log("info", `Bank Marketing: train=${trainData.length} samples, ${trainData[0].features.length} features`);
    return { trainData, testData };
  }

  loadTabularCSV(featuresPath, labelsPath) {
    const parseCSV = (filePath) => {
      const content = fs.readFileSync(filePath, "utf-8");
      const lines = content.trim().split("\n");
      const dataLines = lines.slice(1); // skip header
      return dataLines.map((line) => line.split(",").map(Number));
    };

    const featureRows = parseCSV(featuresPath);
    const labelRows = parseCSV(labelsPath);

    return featureRows.map((features, i) => ({
      features,
      label: labelRows[i][0],
    }));
  }

  async loadSent140Data(participantCount) {
    const dataPath = path.join(this.dataDir, "sent140");
    const trainPath = path.join(dataPath, "train", "all_data.json");
    const testPath = path.join(dataPath, "test", "all_data.json");

    for (const p of [trainPath, testPath]) {
      if (!fs.existsSync(p)) {
        throw new Error(`Sent140 data file not found: ${p}\nPlease run \`npm run download:sent140\` first.`);
      }
    }

    const MAX_VOCAB_SIZE = 10000;
    const MAX_LEN = 25;

    // Load raw JSON
    const trainJson = JSON.parse(fs.readFileSync(trainPath, "utf-8"));
    const testJson = JSON.parse(fs.readFileSync(testPath, "utf-8"));

    // Build vocabulary from training data
    const wordCounts = {};
    for (const user of trainJson.users) {
      const userData = trainJson.user_data[user];
      for (const x of userData.x) {
        const text = x.length > 4 ? x[4] : x[x.length - 1];
        const words = this._splitLine(text.toLowerCase());
        for (const word of words) {
          wordCounts[word] = (wordCounts[word] || 0) + 1;
        }
      }
    }
    const sortedWords = Object.entries(wordCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, MAX_VOCAB_SIZE - 2);
    const vocab = { "<PAD>": 0, "<UNK>": 1 };
    for (const [word] of sortedWords) {
      vocab[word] = Object.keys(vocab).length;
    }
    const vocabSize = Object.keys(vocab).length;
    this.log("info", `Sent140 vocabulary size: ${vocabSize}`);

    // Convert text to word index sequences
    const textToIndices = (text) => {
      const words = this._splitLine(text.toLowerCase());
      const indices = words.slice(0, MAX_LEN).map((w) => (vocab[w] !== undefined ? vocab[w] : vocab["<UNK>"]));
      while (indices.length < MAX_LEN) indices.push(vocab["<PAD>"]);
      return indices;
    };

    // Build flat train samples
    const trainData = [];
    for (const user of trainJson.users) {
      const userData = trainJson.user_data[user];
      for (let i = 0; i < userData.x.length; i++) {
        const text = userData.x[i].length > 4 ? userData.x[i][4] : userData.x[i][userData.x[i].length - 1];
        trainData.push({ indices: textToIndices(text), label: parseInt(userData.y[i]) });
      }
    }

    // Build flat test samples
    const testData = [];
    for (const user of testJson.users) {
      const userData = testJson.user_data[user];
      for (let i = 0; i < userData.x.length; i++) {
        const text = userData.x[i].length > 4 ? userData.x[i][4] : userData.x[i][userData.x[i].length - 1];
        testData.push({ indices: textToIndices(text), label: parseInt(userData.y[i]) });
      }
    }

    // Diagnostic logging
    const labelCounts = trainData.reduce((acc, s) => {
      acc[s.label] = (acc[s.label] || 0) + 1;
      return acc;
    }, {});
    const testLabelCounts = testData.reduce((acc, s) => {
      acc[s.label] = (acc[s.label] || 0) + 1;
      return acc;
    }, {});
    const samplesPerUser = trainJson.users.map((u) => trainJson.user_data[u].x.length);
    const avgSamples = samplesPerUser.reduce((a, b) => a + b, 0) / samplesPerUser.length;
    const minSamples = samplesPerUser.reduce((a, b) => (b < a ? b : a), Infinity);
    const maxSamples = samplesPerUser.reduce((a, b) => (b > a ? b : a), -Infinity);
    this.log("info", `Sent140 dataset stats:`);
    this.log(
      "info",
      `  Train: ${trainData.length} samples, ${trainJson.users.length} users (avg ${avgSamples.toFixed(
        1
      )}/user, min ${minSamples}, max ${maxSamples})`
    );
    this.log("info", `  Test:  ${testData.length} samples, ${testJson.users.length} users`);
    this.log("info", `  Train label balance: neg=${labelCounts[0] || 0}, pos=${labelCounts[1] || 0}`);
    this.log("info", `  Test  label balance: neg=${testLabelCounts[0] || 0}, pos=${testLabelCounts[1] || 0}`);
    this.log("info", `  Vocab size: ${vocabSize}, maxLen: ${MAX_LEN}`);

    // Split train data among participants
    const shuffled = [...trainData].sort(() => 0.5 - Math.random());
    const chunkSize = Math.ceil(shuffled.length / participantCount);
    const participantData = [];
    for (let i = 0; i < participantCount; i++) {
      const start = i * chunkSize;
      participantData.push(shuffled.slice(start, Math.min(start + chunkSize, shuffled.length)));
    }

    return {
      participantData,
      testData,
      sent140Meta: { vocabSize, maxLen: MAX_LEN },
    };
  }

  _splitLine(line) {
    const matches = line.match(/[\w']+|[.,!?;]/g);
    return matches || [];
  }

  async loadCoraData(participantCount) {
    const dataPath = path.join(this.dataDir, "cora");
    const contentPath = path.join(dataPath, "cora.content");
    const citesPath = path.join(dataPath, "cora.cites");

    for (const p of [contentPath, citesPath]) {
      if (!fs.existsSync(p)) {
        throw new Error(`Cora data file not found: ${p}\nPlease run \`npm run download:cora\` first.`);
      }
    }

    // Parse cora.content: <nodeId> <feature0> ... <featureN> <label>
    const contentLines = fs.readFileSync(contentPath, "utf-8").trim().split("\n");
    const nodeIds = [];
    const featuresArray = [];
    const labelsArray = [];

    for (const line of contentLines) {
      const parts = line.trim().split(/\s+/);
      nodeIds.push(parts[0]);
      featuresArray.push(parts.slice(1, -1).map(Number));
      labelsArray.push(parts[parts.length - 1]);
    }

    const numNodes = nodeIds.length;
    const numFeatures = featuresArray[0].length;

    // Node ID -> index map
    const idxMap = new Map();
    nodeIds.forEach((id, idx) => idxMap.set(id.trim(), idx));

    // Encode labels
    const uniqueLabels = [...new Set(labelsArray)].sort();
    const labelMap = new Map();
    uniqueLabels.forEach((label, idx) => labelMap.set(label, idx));
    const encodedLabels = labelsArray.map((l) => labelMap.get(l));
    const numClasses = uniqueLabels.length;

    // Parse cora.cites: <src> <dst>
    const citesLines = fs.readFileSync(citesPath, "utf-8").trim().split("\n");
    const adjData = new Map();

    for (const line of citesLines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 2) {
        const src = idxMap.get(parts[0].trim());
        const dst = idxMap.get(parts[1].trim());
        if (src !== undefined && dst !== undefined) {
          adjData.set(`${src},${dst}`, 1);
          adjData.set(`${dst},${src}`, 1);
        }
      }
    }

    // Add self-loops
    for (let i = 0; i < numNodes; i++) {
      adjData.set(`${i},${i}`, 1);
    }

    // Build dense adjacency matrix and row-normalize
    const adjDense = new Float32Array(numNodes * numNodes);
    for (const key of adjData.keys()) {
      const [i, j] = key.split(",").map(Number);
      adjDense[i * numNodes + j] = 1;
    }

    const rowSums = new Float32Array(numNodes);
    for (let i = 0; i < numNodes; i++) {
      let sum = 0;
      for (let j = 0; j < numNodes; j++) sum += adjDense[i * numNodes + j];
      rowSums[i] = sum;
    }
    for (let i = 0; i < numNodes; i++) {
      if (rowSums[i] > 0) {
        for (let j = 0; j < numNodes; j++) adjDense[i * numNodes + j] /= rowSums[i];
      }
    }

    // Row-normalize features
    const featuresFlat = new Float32Array(numNodes * numFeatures);
    for (let i = 0; i < numNodes; i++) {
      let sum = featuresArray[i].reduce((a, b) => a + b, 0);
      for (let j = 0; j < numFeatures; j++) {
        featuresFlat[i * numFeatures + j] = sum > 0 ? featuresArray[i][j] / sum : 0;
      }
    }

    // Standard Cora splits: train=0..139, val=200..499, test=500..1499
    const idxTrain = Array.from({ length: 140 }, (_, i) => i);
    const idxVal = Array.from({ length: 300 }, (_, i) => i + 200);
    const idxTest = Array.from({ length: 1000 }, (_, i) => i + 500);

    // Split training indices among participants (IID)
    const shuffled = [...idxTrain].sort(() => 0.5 - Math.random());
    const chunkSize = Math.ceil(shuffled.length / participantCount);
    const participantTrainIdx = [];
    for (let i = 0; i < participantCount; i++) {
      const start = i * chunkSize;
      participantTrainIdx.push(shuffled.slice(start, Math.min(start + chunkSize, shuffled.length)));
    }

    this.log("info", `Cora loaded: ${numNodes} nodes, ${numFeatures} features, ${numClasses} classes`);
    this.log("info", `Train: ${idxTrain.length}, Val: ${idxVal.length}, Test: ${idxTest.length}`);

    // Return graph object — participantData holds train index arrays, testData is the graph metadata
    return {
      participantData: participantTrainIdx,
      testData: idxTest,
      graphData: {
        adjFlat: adjDense,
        featuresFlat,
        labels: encodedLabels,
        numNodes,
        numFeatures,
        numClasses,
        idxTrain,
        idxVal,
        idxTest,
      },
    };
  }

  splitDataForParticipants(data, participantCount) {
    const maxSamplesPerParticipant = Math.min(this.maxDataSizePerParticipant, Math.floor(data.length / participantCount));

    if (maxSamplesPerParticipant === 0) {
      throw new Error(
        `Not enough data to distribute among ${participantCount} participants. Minimum 1 sample per participant required.`
      );
    }

    const shuffledData = [...data].sort(() => 0.5 - Math.random());

    const participantData = [];
    let offset = 0;
    for (let i = 0; i < participantCount; i++) {
      const slice = shuffledData.slice(offset, offset + maxSamplesPerParticipant);
      participantData.push(slice);
      offset += maxSamplesPerParticipant;
    }

    participantData.forEach((pData, idx) => {
      this.log("debug", `Participant ${idx + 1} data: ${pData.length} samples`);
    });

    return participantData;
  }
}

module.exports = DataLoader;

const axios = require("axios");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const zlib = require("zlib");

const DATA_DIR = path.join(__dirname, "..", "target", "data");
const TEMP_DIR = path.join(DATA_DIR, "mnist-temp");

const FILES_TO_DOWNLOAD = [
  {
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    md5: "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    name: "train-images-idx3-ubyte",
  },
  {
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    md5: "d53e105ee54ea40749a09fcbcd1e9432",
    name: "train-labels-idx1-ubyte",
  },
  {
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    md5: "9fb629c4189551a2d022fa330f9573f3",
    name: "t10k-images-idx3-ubyte",
  },
  {
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    md5: "ec29112dd5afa0611ce80d1b7f02629c",
    name: "t10k-labels-idx1-ubyte",
  },
];

async function downloadFile(url, dest) {
  console.log(`Downloading ${url} to ${dest}...`);
  const writer = fs.createWriteStream(dest);

  const response = await axios({ url, method: "GET", responseType: "stream" });

  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on("finish", () => {
      console.log(`Download complete: ${path.basename(dest)}`);
      resolve();
    });
    writer.on("error", reject);
  });
}

function verifyMD5(filePath, expectedMd5) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("md5");
    const stream = fs.createReadStream(filePath);
    stream.on("data", (data) => hash.update(data));
    stream.on("end", () => {
      const actualMd5 = hash.digest("hex");
      if (actualMd5 === expectedMd5) {
        console.log(`MD5 verified for ${path.basename(filePath)}.`);
        resolve(true);
      } else {
        console.error(`MD5 mismatch for ${path.basename(filePath)}. Expected: ${expectedMd5}, Got: ${actualMd5}`);
        resolve(false);
      }
    });
    stream.on("error", reject);
  });
}

function decompressGzip(source, dest) {
  return new Promise((resolve, reject) => {
    console.log(`Decompressing ${path.basename(source)}...`);
    const readStream = fs.createReadStream(source);
    const writeStream = fs.createWriteStream(dest);
    const gzip = zlib.createGunzip();

    readStream.pipe(gzip).pipe(writeStream);

    writeStream.on("finish", () => {
      console.log(`Decompressed to ${path.basename(dest)}.`);
      resolve();
    });
    writeStream.on("error", reject);
    readStream.on("error", reject);
    gzip.on("error", reject);
  });
}

function convertIdxToCsv(imagesFile, labelsFile, outputFile) {
  console.log(`Converting to CSV: ${path.basename(outputFile)}...`);
  const imagesBuffer = fs.readFileSync(imagesFile);
  const labelsBuffer = fs.readFileSync(labelsFile);

  const imagesMagic = imagesBuffer.readUInt32BE(0);
  const numImages = imagesBuffer.readUInt32BE(4);
  const numRows = imagesBuffer.readUInt32BE(8);
  const numCols = imagesBuffer.readUInt32BE(12);

  const labelsMagic = labelsBuffer.readUInt32BE(0);
  const numLabels = labelsBuffer.readUInt32BE(4);

  if (imagesMagic !== 2051 || labelsMagic !== 2049) {
    throw new Error("Invalid magic numbers in IDX files.");
  }

  if (numImages !== numLabels) {
    throw new Error(`Mismatch in number of images (${numImages}) and labels (${numLabels}).`);
  }

  const writeStream = fs.createWriteStream(outputFile);

  let imageOffset = 16;
  let labelOffset = 8;
  const imageSize = numRows * numCols;

  for (let i = 0; i < numImages; i++) {
    const label = labelsBuffer.readUInt8(labelOffset++);
    const pixels = [];
    for (let j = 0; j < imageSize; j++) {
      pixels.push(imagesBuffer.readUInt8(imageOffset++));
    }
    writeStream.write(`${label},${pixels.join(",")}\n`);
  }

  return new Promise((resolve, reject) => {
    writeStream.end();
    writeStream.on("finish", () => {
      console.log(`Successfully created ${path.basename(outputFile)}`);
      resolve();
    });
    writeStream.on("error", reject);
  });
}

async function main() {
  try {
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
    }

    if (!fs.existsSync(TEMP_DIR)) {
      fs.mkdirSync(TEMP_DIR, { recursive: true });
    }

    const trainCsvPath = path.join(DATA_DIR, "mnist_train.csv");
    const testCsvPath = path.join(DATA_DIR, "mnist_test.csv");

    if (fs.existsSync(trainCsvPath) && fs.existsSync(testCsvPath)) {
      console.log("MNIST CSV dataset already exists. Skipping download.");
      return;
    }

    // Download and extract files
    for (const file of FILES_TO_DOWNLOAD) {
      const gzPath = path.join(TEMP_DIR, `${file.name}.gz`);
      const outPath = path.join(TEMP_DIR, file.name);

      if (!fs.existsSync(outPath)) {
        if (!fs.existsSync(gzPath)) {
          await downloadFile(file.url, gzPath);
          const isValid = await verifyMD5(gzPath, file.md5);
          if (!isValid) {
            throw new Error(`Downloaded file ${file.name}.gz is corrupted.`);
          }
        }
        await decompressGzip(gzPath, outPath);
        if (fs.existsSync(gzPath)) {
          fs.unlinkSync(gzPath);
          console.log(`Cleaned up ${path.basename(gzPath)}.`);
        }
      } else {
        console.log(`${file.name} already decompressed. Skipping.`);
      }
    }

    // Convert to CSV
    if (!fs.existsSync(trainCsvPath)) {
      await convertIdxToCsv(
        path.join(TEMP_DIR, "train-images-idx3-ubyte"),
        path.join(TEMP_DIR, "train-labels-idx1-ubyte"),
        trainCsvPath
      );
    }
    
    if (!fs.existsSync(testCsvPath)) {
      await convertIdxToCsv(
        path.join(TEMP_DIR, "t10k-images-idx3-ubyte"),
        path.join(TEMP_DIR, "t10k-labels-idx1-ubyte"),
        testCsvPath
      );
    }

    // Cleanup temp directory
    fs.rmSync(TEMP_DIR, { recursive: true, force: true });
    console.log("Cleaned up temporary idx files.");

    console.log("MNIST dataset preparation complete.");
  } catch (error) {
    console.error("An error occurred:", error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

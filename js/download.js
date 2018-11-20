const fs = require("fs");
const request = require("request-promise-native");
const { promisify } = require("util");
const writeFile = promisify(fs.writeFile);

async function downloadData() {
  let results = [];
  let currentPage = 0;
  let totalPages = null;
  while (totalPages === null || currentPage <= totalPages) {
    const response = await request({
      method: "GET",
      uri: "https://glazy.org/api/search",
      json: true,
      qs: {
        photo: true,
        base_type: 460,
        p: currentPage
      }
    });
    const {
      data,
      meta: { pagination }
    } = response;
    totalPages = pagination.total_pages;
    currentPage += 1;
    results = [...results, ...data];
  }
  const ids = {};
  results.map(d => d.id).forEach(d => {
    ids[d] = true;
  });
  console.log(`unique IDS: ${Object.keys(ids).length}`);
  await writeFile("data.json", JSON.stringify(results, null, 2));
}

async function downloadImages() {
  const data = require("./data.json");
  let i = 0;
  const BATCH = 40;
  while (i < data.length) {
    const j = Math.min(data.length, i + BATCH);
    const promises = data.slice(i, j).map(downloadImage);
    await Promise.all(promises);
    console.log(`downloads: ${downloads}`);
    console.log(`total: ${total}`);
    i = j;
  }
  console.log("Done");
  console.log(`Unique filenames: ${Object.keys(names).length}`);
}

let downloads = 0;
let total = 0;
const names = {};
async function downloadImage(glazeData) {
  const {
    selectedImage: { filename }
  } = glazeData;
  const exists = await checkFileExists(`images/${filename}`);
  names[filename] = true;
  if (exists) {
    total += 1;
    return;
  }
  const dir = filename.split(".")[0].slice(-2);
  const image = await request({
    method: "GET",
    encoding: "binary",
    uri: `https://glazy.org/storage/uploads/recipes/${dir}/s_${filename}`
  });
  await writeFile(`images/${filename}`, image, "binary");
  downloads += 1;
  total += 1;
}

function checkFileExists(filepath) {
  return new Promise((resolve, reject) => {
    fs.access(filepath, fs.R_OK, error => {
      resolve(!error);
    });
  });
}

async function downloadAll() {
  await downloadData();
  await downloadImages();
}

downloadAll();

import React, { Component } from "react";
import logo from "./logo.svg";
import "./App.css";
import map from "./som_winners.json";
import data from "./glaze_data.json";
const remapped = {};
data.forEach((e, i) => {
  const [row, col] = map.winners[i];
  const image = e.selectedImage.filename;
  if (remapped[row]) {
    if (remapped[row][col]) {
      remapped[row][col].push(image);
    } else {
      remapped[row][col] = [image];
    }
  } else {
    remapped[row] = {
      [col]: [image]
    };
  }
});

const SERVER = process.env.SERVER;

class App extends Component {
  state = {
    state: "notAsked",
    currentId: null
  };
  async getNext() {
    fetch(`${SERVER}`);
  }
  renderGlazesAt(i, j) {
    const glazes = remapped[i] && remapped[i][j];
    if (glazes) {
      return glazes.map(g => (
        <div className="Glaze">
          <img className="Glaze-Image" src={`images/${g}`} />
        </div>
      ));
    } else {
      return null;
    }
  }
  renderMappedGlazes() {
    return Array(50)
      .fill(undefined)
      .map((x, i) => (
        <div className="Glaze-Row">
          {Array(50)
            .fill(undefined)
            .map((x, j) => (
              <div className="Glaze-Cell">{this.renderGlazesAt(i, j)}</div>
            ))}
        </div>
      ));
  }
  render() {
    return <div className="App">{this.renderMappedGlazes()}</div>;
  }
}

export default App;

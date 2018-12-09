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
    zoom: 100,
    isDragging: false,
    initialX: null,
    initialY: null,
    top: 0,
    left: 0,
    state: "notAsked",
    currentId: null
  };
  async getNext() {
    fetch(`${SERVER}`);
  }
  handleAppRef = ref => {
    if (ref) {
      this.ref = ref;
      ref.addEventListener("wheel", this.handleWheel);
      ref.addEventListener("mousedown", this.handleMouseDown);
      ref.addEventListener("mouseup", this.handleMouseUp);
      ref.addEventListener("mousemove", this.handleMouseMove);
    } else {
      ref.removeEventListener("wheel", this.handleWheel);
      ref.removeEventListener("mouseDown", this.handleMouseDown);
      ref.removeEventListener("mouseUp", this.handleMouseUp);
      ref.removeEventListener("mouseMove", this.handleMouseMove);
      this.ref = ref;
    }
  };
  handleWheel = e => {
    this.setState(s => ({
      ...s,
      zoom: (s.zoom -= e.deltaY)
    }));
  };
  handleMouseDown = e => {
    this.setState(s => ({
      ...s,
      isDragging: true,
      initialX: e.clientX - s.left,
      initialY: e.clientY - s.top
    }));
  };
  handleMouseMove = e => {
    this.setState(s => {
      if (!s.isDragging) {
        return s;
      }
      return {
        ...s,
        left: e.clientX - s.initialX,
        top: e.clientY - s.initialY
      };
    });
  };
  handleMouseUp = e => {
    this.setState(s => ({
      ...s,
      isDragging: false
    }));
  };
  renderGlazesAt(i, j) {
    const glazes = remapped[i] && remapped[i][j];
    if (glazes) {
      const height = `${100 / Math.sqrt(glazes.length)}%`;
      const width = height;
      return glazes.map(g => (
        <div className="Glaze" style={{ height, width }}>
          <img draggable={false} className="Glaze-Image" src={`images/${g}`} />
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
    const { zoom, top, left } = this.state;
    const width = `${zoom}vw`;
    const height = `${zoom}vh`;
    console.log(`(${left}, ${top}) x ${zoom}`);
    return (
      <div
        ref={this.handleAppRef}
        className="App"
        style={{ height, width, top, left }}
      >
        {this.renderMappedGlazes()}
      </div>
    );
  }
}

export default App;

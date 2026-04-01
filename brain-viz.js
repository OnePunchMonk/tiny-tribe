/**
 * 3D Brain visualization using Three.js on fsaverage4 mesh.
 *
 * Renders a brain surface mesh colored by predicted vertex activations.
 */

import * as THREE from "three";

// fsaverage4: 2562 vertices per hemisphere, 5124 total
const FSAVERAGE4_VERTICES = 5124;

export class BrainVisualization {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      45,
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 0, 250);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    // Lighting
    this.scene.add(new THREE.AmbientLight(0x404040, 2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(100, 100, 100);
    this.scene.add(dirLight);
    const backLight = new THREE.DirectionalLight(0x6366f1, 0.5);
    backLight.position.set(-100, -50, -100);
    this.scene.add(backLight);

    // Brain mesh (placeholder sphere until real mesh loads)
    this.brainMesh = null;
    this.activationColors = null;

    this._createPlaceholderBrain();
    this._setupInteraction();
    this._animate();
  }

  _createPlaceholderBrain() {
    // Use an icosphere as a stand-in for fsaverage4
    // The real mesh would be loaded from a .json or .ply file
    const geometry = new THREE.IcosahedronGeometry(80, 5);

    // Vertex colors
    const colors = new Float32Array(geometry.attributes.position.count * 3);
    colors.fill(0.5);
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      shininess: 30,
      side: THREE.DoubleSide,
    });

    this.brainMesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.brainMesh);
    this.activationColors = colors;
  }

  /**
   * Load an actual fsaverage4 mesh from a JSON file.
   * @param {string} url - URL to mesh JSON with {vertices: [...], faces: [...]}
   */
  async loadMesh(url) {
    try {
      const resp = await fetch(url);
      const data = await resp.json();

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(data.vertices, 3)
      );
      geometry.setIndex(data.faces);
      geometry.computeVertexNormals();

      const colors = new Float32Array(data.vertices.length); // vertices.length / 3 * 3
      colors.fill(0.5);
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        shininess: 30,
        side: THREE.DoubleSide,
      });

      this.scene.remove(this.brainMesh);
      this.brainMesh = new THREE.Mesh(geometry, material);
      this.scene.add(this.brainMesh);
      this.activationColors = colors;
    } catch (e) {
      console.warn("Could not load brain mesh, using placeholder:", e);
    }
  }

  /**
   * Update brain coloring from vertex activation values.
   * @param {Float32Array} activations - (n_vertices,) activation values
   */
  updateActivations(activations) {
    if (!this.brainMesh) return;

    const colors = this.brainMesh.geometry.attributes.color;
    const nVertices = colors.count;

    // Normalize activations to [0, 1]
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < activations.length; i++) {
      if (activations[i] < min) min = activations[i];
      if (activations[i] > max) max = activations[i];
    }
    const range = max - min || 1;

    // Map to color (blue → white → red diverging colormap)
    for (let i = 0; i < nVertices && i < activations.length; i++) {
      const t = (activations[i] - min) / range; // 0..1
      const [r, g, b] = this._divergingColor(t);
      colors.setXYZ(i, r, g, b);
    }

    colors.needsUpdate = true;
  }

  /**
   * Blue → White → Red diverging colormap.
   * @param {number} t - Value in [0, 1] where 0.5 = neutral
   */
  _divergingColor(t) {
    if (t < 0.5) {
      // Blue to white
      const s = t * 2; // 0..1
      return [s, s, 1.0];
    } else {
      // White to red
      const s = (t - 0.5) * 2; // 0..1
      return [1.0, 1.0 - s, 1.0 - s];
    }
  }

  _setupInteraction() {
    let isDragging = false;
    let prevX = 0, prevY = 0;

    this.canvas.addEventListener("mousedown", (e) => {
      isDragging = true;
      prevX = e.clientX;
      prevY = e.clientY;
    });

    window.addEventListener("mouseup", () => { isDragging = false; });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging || !this.brainMesh) return;
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      this.brainMesh.rotation.y += dx * 0.005;
      this.brainMesh.rotation.x += dy * 0.005;
      prevX = e.clientX;
      prevY = e.clientY;
    });

    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.camera.position.z += e.deltaY * 0.1;
      this.camera.position.z = Math.max(100, Math.min(500, this.camera.position.z));
    });
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    if (this.brainMesh && !this._userInteracting) {
      this.brainMesh.rotation.y += 0.002;
    }
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}

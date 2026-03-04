(function () {
  class SplitSceneViewer {
    constructor(containerId, pcdPath, pcaPath) {
      this.containerId = containerId;

      this.config = {
        camera: { fov: 45, near: 1, far: 2000, position: { x: 0, y: 0, z: 5 } },
        controls: { minDistance: 0.0, maxDistance: 1000 },
        material: { size: 1.25 },
        scene: { backgroundColor: 0xffffff },
      };

      this.splitRatio = 0.5; // 0~1
      this.draggingSplitter = false;

      this.pcdPath = pcdPath;
      this.pcaPath = pcaPath;

      this.pcdPoints = null; // layer 0
      this.pcaPoints = null; // layer 1

      this.init();
    }

    getWrapperEl() {
      const container = document.getElementById(this.containerId);
      if (!container) throw new Error(`container #${this.containerId} not found`);
      return container.parentNode;
    }

    getCanvasDimensions() {
        const wrapper = this.getWrapperEl();
        const rect = wrapper.getBoundingClientRect();
        const canvasWidth = Math.max(1, Math.floor(rect.width));
        const canvasHeight = Math.max(1, Math.floor(rect.height));
        return { canvasWidth, canvasHeight };
    }

    createCamera(w, h) {
      const cam = new THREE.PerspectiveCamera(this.config.camera.fov, w / h, this.config.camera.near, this.config.camera.far);
      cam.position.set(this.config.camera.position.x, this.config.camera.position.y, this.config.camera.position.z);
      return cam;
    }

    createScene() {
      const s = new THREE.Scene();
      s.background = new THREE.Color(this.config.scene.backgroundColor);
      return s;
    }

    createRenderer(w, h) {
      const r = new THREE.WebGLRenderer({ antialias: true });
      r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      r.setSize(w, h);

      const container = document.getElementById(this.containerId);
      container.innerHTML = "";
      container.appendChild(r.domElement);
      return r;
    }

    createControls(camera, domElement) {
      const c = new THREE.OrbitControls(camera, domElement);
      c.minDistance = this.config.controls.minDistance;
      c.maxDistance = this.config.controls.maxDistance;
      c.addEventListener("change", () => this.render());
      return c;
    }

    createCustomPointMaterial() {
      return new THREE.ShaderMaterial({
        uniforms: { pointSize: { value: this.config.material.size } },
        vertexShader: `
          uniform float pointSize;
          varying vec3 vColor;
          void main() {
            vColor = color;
            gl_PointSize = pointSize;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          varying vec3 vColor;
          void main() {
            gl_FragColor = vec4(vColor, 1.0);
          }
        `,
        vertexColors: true,
      });
    }

    addAxesHelper() {
      const axes = new THREE.AxesHelper(5);
      axes.rotation.z = Math.PI / 2;
      axes.layers.enable(0);
      axes.layers.enable(1);
      this.scene.add(axes);
    }

    createSplitterUI() {
      const wrapper = this.getWrapperEl();

      const old = wrapper.querySelector(".sv-splitter");
      if (old) old.remove();

      const splitter = document.createElement("div");
      splitter.className = "sv-splitter";
      const handle = document.createElement("div");
      handle.className = "sv-splitter-handle";
      splitter.appendChild(handle);
      wrapper.appendChild(splitter);

      this.splitter = splitter;

      const updatePos = () => {
        splitter.style.left = `${this.splitRatio * 100}%`;
      };
      updatePos();

      splitter.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.draggingSplitter = true;
        this.controls.enabled = false;
        splitter.setPointerCapture?.(e.pointerId);
      });

      window.addEventListener("pointermove", (e) => {
        if (!this.draggingSplitter) return;
        const rect = wrapper.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const ratio = x / Math.max(1, rect.width);
        this.splitRatio = Math.min(0.95, Math.max(0.05, ratio));
        updatePos();
        this.render();
      });

      window.addEventListener("pointerup", (e) => {
        if (!this.draggingSplitter) return;
        this.draggingSplitter = false;
        this.controls.enabled = true;
        splitter.releasePointerCapture?.(e.pointerId);
      });
    }

    attachControlButtonListeners() {
      const wrapper = this.getWrapperEl();
      wrapper.querySelector(".sv-zoom-in-button")?.addEventListener("click", () => this.simulateZoom(true));
      wrapper.querySelector(".sv-zoom-out-button")?.addEventListener("click", () => this.simulateZoom(false));
      wrapper.querySelector(".sv-reset-button")?.addEventListener("click", () => this.resetView());
    }

    simulateZoom(isZoomIn) {
      const event = new WheelEvent("wheel", {
        deltaY: isZoomIn ? -100 : 100,
        bubbles: true,
        cancelable: true,
      });
      this.renderer.domElement.dispatchEvent(event);
    }

    resetView() {
      this.camera.position.set(this.config.camera.position.x, this.config.camera.position.y, this.config.camera.position.z);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
      this.render();
    }

    adjustCanvasSize() {
      const { canvasWidth, canvasHeight } = this.getCanvasDimensions();
      this.camera.aspect = canvasWidth / canvasHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(canvasWidth, canvasHeight);
      this.render();
    }

    loadPLYGeometry(path) {
      return new Promise((resolve, reject) => {
        const loader = new THREE.PLYLoader();
        loader.load(path, resolve, undefined, reject);
      });
    }

    disposePoints(points) {
      if (!points) return;
      this.scene.remove(points);
      points.geometry?.dispose();
      points.material?.dispose();
    }

    async loadPointCloudPair(pcdPath, pcaPath) {
      this.disposePoints(this.pcdPoints);
      this.disposePoints(this.pcaPoints);
      this.pcdPoints = null;
      this.pcaPoints = null;

      const [pcdGeo, pcaGeo] = await Promise.all([
        this.loadPLYGeometry(pcdPath),
        this.loadPLYGeometry(pcaPath),
      ]);

      pcdGeo.computeBoundingBox();
      pcaGeo.computeBoundingBox();

      // 统一中心，保证两份点云对齐
      const unionBox = pcdGeo.boundingBox.clone().union(pcaGeo.boundingBox);
      const center = new THREE.Vector3();
      unionBox.getCenter(center);

      pcdGeo.translate(-center.x, -center.y, -center.z);
      pcaGeo.translate(-center.x, -center.y, -center.z);

      const pcd = new THREE.Points(pcdGeo, this.createCustomPointMaterial());
      const pca = new THREE.Points(pcaGeo, this.createCustomPointMaterial());

      pcd.layers.set(0);
      pca.layers.set(1);

      this.scene.add(pcd);
      this.scene.add(pca);

      this.pcdPoints = pcd;
      this.pcaPoints = pca;

      this.render();
    }

    render() {
      if (!this.renderer || !this.camera) return;

      const size = this.renderer.getSize(new THREE.Vector2());
      const w = Math.max(1, Math.floor(size.x));
      const h = Math.max(1, Math.floor(size.y));
      const splitX = Math.max(1, Math.min(w - 1, Math.floor(w * this.splitRatio)));

      this.renderer.setScissorTest(false);
      this.renderer.clear(true, true, true);

      this.renderer.setScissorTest(true);
      this.renderer.setViewport(0, 0, w, h);

      this.camera.layers.set(0);
      this.renderer.setScissor(0, 0, splitX, h);
      this.renderer.render(this.scene, this.camera);

      this.camera.layers.set(1);
      this.renderer.setScissor(splitX, 0, w - splitX, h);
      this.renderer.render(this.scene, this.camera);

      this.renderer.setScissorTest(false);
    }

    async init() {
      const { canvasWidth, canvasHeight } = this.getCanvasDimensions();
      this.camera = this.createCamera(canvasWidth, canvasHeight);
      this.scene = this.createScene();
      this.renderer = this.createRenderer(canvasWidth, canvasHeight);
      this.controls = this.createControls(this.camera, this.renderer.domElement);

      this.createSplitterUI();
      this.attachControlButtonListeners();

      await this.loadPointCloudPair(this.pcdPath, this.pcaPath);

      window.addEventListener("resize", () => this.adjustCanvasSize());

      const tick = () => {
        requestAnimationFrame(tick);
        this.controls.update();
      };
      tick();
    }
  }

  let splitViewer = null;

  function initViewers() {
    splitViewer = new SplitSceneViewer(
      "hk-canvas-split",
      "https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/hk/hk_1/origin.ply",
      "https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/hk/hk_1/pca.ply"
    );
  }

  window.loadHKModelPair = async function (name) {
    if (!splitViewer) return;

    const pcdPath = `https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/hk/hk_${name}/origin.ply`;
    const pcaPath = `https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/hk/hk_${name}/pca.ply`;

    await splitViewer.loadPointCloudPair(pcdPath, pcaPath);
    splitViewer.resetView();
  };

  document.addEventListener("DOMContentLoaded", initViewers);
})();

(function () {
    class SceneViewer {
        constructor(containerId, pointCloudPath, manager) {
            this.containerId = containerId;
            this.pointCloudPath = pointCloudPath;
            this.manager = manager;

            this.config = {
                camera: {
                    fov: 45,
                    near: 1,
                    far: 2000,
                    position: { x: 0, y: 0, z: 5 },
                },
                controls: {
                    minDistance: 0.,
                    maxDistance: 1000,
                },
                material: {
                    size: 1.25,
                    vertexColors: true,
                },
                scene: {
                    backgroundColor: 0xffffff,
                },
            };

            this.init();
        }

        getCanvasDimensions() {
            const canvasContainer = document.querySelector(".sv-canvas-container");
            const canvasWrapper = document.querySelector(".sv-canvas-wrapper");

            const wrapperStyle = window.getComputedStyle(canvasWrapper);
            const padding =
                parseFloat(wrapperStyle.paddingLeft) +
                parseFloat(wrapperStyle.paddingRight);
            const border =
                parseFloat(wrapperStyle.borderLeftWidth) +
                parseFloat(wrapperStyle.borderRightWidth);
            const gap = parseFloat(window.getComputedStyle(canvasContainer).gap);

            const wrapperPaddingBorder = padding + border;
            const totalPadding = wrapperPaddingBorder * 2 + gap * 1;

            const canvasWidth = (canvasContainer.clientWidth - totalPadding) / 2;
            const canvasHeight = canvasWidth * 1;

            return { canvasWidth, canvasHeight };
        }

        createCamera(width, height) {
            const camera = new THREE.PerspectiveCamera(
                this.config.camera.fov,
                width / height,
                this.config.camera.near,
                this.config.camera.far
            );
            camera.position.set(
                this.config.camera.position.x,
                this.config.camera.position.y,
                this.config.camera.position.z
            );
            return camera;
        }

        createScene() {
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(this.config.scene.backgroundColor);
            return scene;
        }

        createRenderer(width, height) {
            const renderer = new THREE.WebGLRenderer({ antialias: true }); // Enable antialiasing
            renderer.setSize(width, height);
            document.getElementById(this.containerId).appendChild(renderer.domElement);
            return renderer;
        }

        loadPointCloud() {
            console.log("Loading point cloud from:", this.pointCloudPath);
            const loader = new THREE.PLYLoader();
            loader.load(
                this.pointCloudPath,
                (geometry) => {
                    console.log('PLY file loaded successfully', geometry);
        
                    geometry.computeBoundingBox();
                    const bbox = geometry.boundingBox;
                    const center = new THREE.Vector3();
                    bbox.getCenter(center);
        
                    geometry.translate(-center.x, -center.y, -center.z);
        
                    const material = this.createCustomPointMaterial();
                    const points = new THREE.Points(geometry, material);
                    this.scene.add(points);
        
                    // this.camera.position.set(center.x, center.y, center.z + 3);
                    // this.controls.target.set(center.x, center.y, center.z);
                    // this.controls.update();
        
                    this.render();
                },
                undefined,
                (error) => {
                    console.error('Error loading PLY file:', error);
                }
            );
        }
        createCustomPointMaterial() {
            return new THREE.ShaderMaterial({
                uniforms: {
                    pointSize: { value: this.config.material.size },
                },
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

        createControls(camera, domElement) {
            const controls = new THREE.OrbitControls(camera, domElement);
            controls.minDistance = this.config.controls.minDistance;
            controls.maxDistance = this.config.controls.maxDistance;
            return controls;
        }

        addAxesHelper() {
            const axesHelper = new THREE.AxesHelper(5);
            axesHelper.rotation.z = Math.PI / 2;
            this.scene.add(axesHelper);
        }

        init() {
            const { canvasWidth, canvasHeight } = this.getCanvasDimensions();

            this.camera = this.createCamera(canvasWidth, canvasHeight);
            this.scene = this.createScene();
            this.renderer = this.createRenderer(canvasWidth, canvasHeight);

            const cursor = document.createElement('div');
            cursor.className = 'sv-interactive-cursor';
            document.getElementById(this.containerId).parentNode.appendChild(cursor);
            this.cursor = cursor;

            this.renderer.domElement.addEventListener('pointerdown', () => {
                this.manager.hideAllCursors();
            });

            // this.addAxesHelper();
            this.loadPointCloud();

            this.controls = this.createControls(this.camera, this.renderer.domElement);
            this.controls.addEventListener("change", () =>
                this.manager.synchronizeControls(this)
            );

            this.attachControlButtonListeners();
        }

        attachControlButtonListeners() {
            const wrapper = document.getElementById(this.containerId).parentNode;
            const zoomInButton = wrapper.querySelector(".sv-zoom-in-button");
            const zoomOutButton = wrapper.querySelector(".sv-zoom-out-button");
            const resetButton = wrapper.querySelector(".sv-reset-button");

            zoomInButton.addEventListener("click", () => this.simulateZoom(true));
            zoomOutButton.addEventListener("click", () => this.simulateZoom(false));
            resetButton.addEventListener("click", () => this.resetView());
        }

        simulateZoom(isZoomIn) {
            const event = new WheelEvent("wheel", {
                deltaY: isZoomIn ? -100 : 100,
                bubbles: true,
                cancelable: true,
            });
            this.renderer.domElement.dispatchEvent(event);
        }

        adjustCanvasSize() {
            const { canvasWidth, canvasHeight } = this.getCanvasDimensions();

            this.camera.aspect = canvasWidth / canvasHeight;
            this.camera.updateProjectionMatrix();

            this.renderer.setSize(canvasWidth, canvasHeight);
        }

        render() {
            this.renderer.render(this.scene, this.camera);
        }

        resetView() {
            this.camera.position.set(
                this.config.camera.position.x,
                this.config.camera.position.y,
                this.config.camera.position.z
            );
            this.controls.target.set(0, 0, 0);
            this.controls.update();
            this.render();
        }
    }

    class ViewerManager {
        constructor() {
            this.viewers = [];
            this.syncing = false;
            this.cursorsHidden = false;
        }

        addViewer(viewer) {
            this.viewers.push(viewer);
        }

        synchronizeControls(activeViewer) {
            if (this.syncing) return;
            this.syncing = true;

            const { camera, controls } = activeViewer;

            this.viewers.forEach((viewer) => {
                if (viewer !== activeViewer) {
                    viewer.camera.position.copy(camera.position);
                    viewer.camera.rotation.copy(camera.rotation);
                    viewer.controls.target.copy(controls.target);
                    viewer.controls.update();
                }
                viewer.render();
            });

            this.syncing = false;
        }

        adjustCanvasSizes() {
            this.viewers.forEach((viewer) => viewer.adjustCanvasSize());
        }

        renderAll() {
            this.viewers.forEach((viewer) => viewer.render());
        }

        getCanvasDimensions() {
            const canvasContainer = document.querySelector(".sv-canvas-container");
            const canvasWrapper = document.querySelector(".sv-canvas-wrapper");

            const wrapperStyle = window.getComputedStyle(canvasWrapper);
            const padding =
                parseFloat(wrapperStyle.paddingLeft) +
                parseFloat(wrapperStyle.paddingRight);
            const border =
                parseFloat(wrapperStyle.borderLeftWidth) +
                parseFloat(wrapperStyle.borderRightWidth);
            const gap = parseFloat(window.getComputedStyle(canvasContainer).gap);

            const wrapperPaddingBorder = padding + border;
            const totalPadding = wrapperPaddingBorder * this.viewers.length + gap * (this.viewers.length - 1);

            const canvasWidth = (canvasContainer.clientWidth - totalPadding) / this.viewers.length;
            const canvasHeight = canvasWidth * 1;

            return { canvasWidth, canvasHeight };
        }

        hideAllCursors() {
            if (this.cursorsHidden) return;
            this.cursorsHidden = true;

            this.viewers.forEach(viewer => {
                if (viewer.cursor) {
                    viewer.cursor.classList.add('hidden');
                }
            });
        }
    }

    function init() {
        const manager = new ViewerManager();

        const pcdViewer = new SceneViewer(
            "manipulation-canvas-pcd",
            "https://raw.githubusercontent.com/Pointcept/Assets/main/concerto/demos/ply/1/pcd.ply",
            manager
        );
        const pcaViewer = new SceneViewer(
            "manipulation-canvas-pca",
            "https://raw.githubusercontent.com/Pointcept/Assets/main/concerto/demos/ply/1/pca.ply",
            manager
        );

        manager.addViewer(pcdViewer);
        manager.addViewer(pcaViewer);

        window.addEventListener("resize", () => {
            manager.adjustCanvasSizes();
            manager.renderAll();
        });

        animate(manager);
    }

    function animate(manager) {
        requestAnimationFrame(() => animate(manager));
        manager.viewers.forEach((viewer) => viewer.controls.update());
    }

    // document.addEventListener("DOMContentLoaded", init);

    let manager, pcdViewer, pcaViewer;

    function initViewers() {
        manager = new ViewerManager();

        pcdViewer = new SceneViewer("manipulation-canvas-pcd", "https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/manipulation/manipulation_4/origin.ply", manager);
        pcaViewer = new SceneViewer("manipulation-canvas-pca", "https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/manipulation/manipulation_4/pca.ply", manager);

        manager.addViewer(pcdViewer);
        manager.addViewer(pcaViewer);

        window.addEventListener("resize", () => {
            manager.adjustCanvasSizes();
            manager.renderAll();
        });

        animate(manager);
    }

    window.loadManipulationModelPair = function(name) {
        // const pcdPath = `ply/${name}/pcd.ply`;
        // const pcaPath = `ply/${name}/pca.ply`;
        const pcdPath = `https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/manipulation/manipulation_${name}/origin.ply`;
        const pcaPath = `https://raw.githubusercontent.com/Pointcept/Assets/main/utonia/demos/ply/manipulation/manipulation_${name}/pca.ply`;
      
        console.log("Loading new model pair:", name);
      
        pcdViewer.scene.clear();
        pcaViewer.scene.clear();
      
        pcdViewer.pointCloudPath = pcdPath;
        pcaViewer.pointCloudPath = pcaPath;
      
        pcdViewer.loadPointCloud();
        pcaViewer.loadPointCloud();
      
        pcdViewer.resetView();
        pcaViewer.resetView();
      };

    function animate(manager) {
        requestAnimationFrame(() => animate(manager));
        manager.viewers.forEach((viewer) => viewer.controls.update());
    }

    document.addEventListener("DOMContentLoaded", initViewers);

})();
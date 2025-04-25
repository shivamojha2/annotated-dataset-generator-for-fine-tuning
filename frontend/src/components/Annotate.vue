<template>
  <div class="annotation-viewer-app">
    <!-- Always visible header section -->
    <header class="app-header">
      <h1 class="app-title">Image Annotation Viewer</h1>
      <div class="query-summary">
        <div class="query-card">
          <span class="query-label">Query:</span>
          <span class="query-text">{{ $route.query.query }}</span>
        </div>
        <div class="annotations-summary">
          <div 
            v-for="(option, idx) in selectedAnnotationTypes" 
            :key="idx"
            class="annotation-tag"
            :class="{ active: activeAnnotations.includes(option) }"
          >
            {{ option }}
          </div>
        </div>
      </div>
    </header>

    <!-- Initial annotation selection phase -->
    <div v-if="selectionPhase" class="selection-container">
      <div class="selection-content">
        <h2 class="selection-heading">Select Annotation Types</h2>
        <p class="selection-instruction">Choose the types of annotations you want to apply to your images:</p>
        
        <div class="annotation-options">
          <div 
            v-for="(option, index) in annotationOptions" 
            :key="index"
            class="annotation-option"
            :class="{ selected: selectedAnnotationTypes.includes(option) }"
            @click="toggleAnnotationSelection(option)"
          >
            <div class="option-icon" :class="getAnnotationIcon(option)"></div>
            <div class="option-details">
              <span class="option-name">{{ option }}</span>
              <span class="option-description">{{ getAnnotationDescription(option) }}</span>
            </div>
            <div class="selection-indicator"></div>
          </div>
        </div>
        
        <div class="selection-actions">
          <button 
            class="proceed-button" 
            :disabled="selectedAnnotationTypes.length === 0"
            @click="proceedToLoading"
          >
            Generate Annotations
          </button>
        </div>
      </div>
    </div>

    <!-- Loading state when images are being fetched -->
    <div v-else-if="isLoading" class="loading-container">
      <div class="loading-spinner"></div>
      <p class="loading-text">Preparing your annotated images...</p>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: `${loadingProgress}%` }"></div>
      </div>
    </div>

    <!-- Image gallery view -->
    <div v-else class="image-gallery-container">
      <!-- Annotation filters sticky bar -->
      <div class="annotation-filter-bar">
        <span class="filter-title">Show Annotations:</span>
        <div class="filter-options">
          <div 
            v-for="(option, idx) in selectedAnnotationTypes" 
            :key="idx"
            class="filter-chip"
            :class="{ active: activeAnnotations.includes(option) }"
            @click="toggleAnnotationType(option)"
          >
            <span class="filter-icon" :class="getAnnotationIcon(option)"></span>
            {{ option }}
          </div>
        </div>
      </div>

      <!-- Image grid -->
      <div class="image-grid">
        <div 
          v-for="(image, index) in annotatedImages" 
          :key="index"
          class="image-card"
          @click="selectImage(index)"
        >
        <div class="image-container">
          <img :src="image.url" :alt="`Annotated image ${index + 1}`" class="image-thumbnail" @load="updateImageDimensions($event, index)">
          <div class="annotation-overlays">
            <!-- Overlays for each active annotation type -->
            <div 
              v-for="type in activeAnnotations" 
              :key="`${index}-${type}`"
              class="annotation-overlay"
              :class="[`annotation-${type.toLowerCase().replace(' ', '-')}`]"
            >
              <!-- Bounding Boxes - Now handling multiple boxes -->
               <!-- <p>{{ displayedImageDimensions[index] }}</p> -->
              <div
                v-if="type === 'Bounding Boxes' && displayedImageDimensions[index] && image.bboxes && image.shape"
                class="bounding-boxes-container"
              >
                <div 
                  v-for="(bbox, bboxIndex) in image.bboxes" 
                  :key="`bbox-${index}-${bboxIndex}`"
                  class="bounding-box"
                  :style="calculateBoundingBoxStyle(index, bboxIndex)"
                ></div>
              </div>
              
              <!-- Segmentation Masks - Now handling multiple masks -->
              <div 
                v-if="type === 'Segmentation' && displayedImageDimensions[index] && image.mask_urls && image.mask_urls.length > 0"
                class="segmentation-mask-container"
              >
                <img
                  v-for="(maskUrl, maskIndex) in image.mask_urls"
                  :key="`mask-${index}-${maskIndex}`"
                  :src="maskUrl"  
                  class="segmentation-mask" 
                  :style="calculateMaskStyle(index)"
                />
              </div>
              
              <!-- Other annotation types -->
              <div v-if="type === 'Keypoints'" class="keypoints-placeholder"></div>             </div>
            </div>
            <div class="image-metadata">
              <span class="image-count">{{ getImageAnnotationCount(image) }} annotations</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Modal for zoomed image view -->
    <transition name="modal">
      <div v-if="selectedImageIndex !== null" class="modal-overlay" @click.self="closeImageModal">
        <div class="modal-content">
          <button class="close-modal-btn" @click="closeImageModal">×</button>
          
          <div class="modal-image-container">
          <!--
          <div class="zoom-controls">
            <button @click="changeZoom(-10)" class="zoom-btn">-</button>
            <span class="zoom-level">{{ zoomLevel }}%</span>
            <button @click="changeZoom(10)" class="zoom-btn">+</button>
            <button @click="resetZoom" class="zoom-btn reset">Reset</button>
          </div>
          -->

          <div class="zoomed-image-wrapper" ref="zoomedImageWrapper" :style="{ transform: `scale(${zoomLevel / 100})` }">
              <!-- Image -->
              <img 
                v-if="selectedImage && selectedImage.url" 
                :src="selectedImage.url" 
                :alt="selectedImage.description || 'Zoomed image'" 
                class="zoomed-image" 
              />

              <!-- Bounding Boxes -->
              <div 
                v-if="isZoomedImageReady && zoomedAnnotations.includes('Bounding Boxes') && selectedImage.bboxes && selectedImage.shape"
                class="modal-bounding-boxes-container"
              >
                <div 
                  v-for="(bbox, bboxIndex) in selectedImage.bboxes" 
                  :key="`modal-bbox-${bboxIndex}`"
                  class="modal-bounding-box"
                  :style="calculateModalBoundingBoxStyle(bboxIndex)"
                ></div>
              </div>

              <!-- Segmentation Masks -->
              <div 
                v-if="zoomedAnnotations.includes('Segmentation') && selectedImage.mask_urls && selectedImage.mask_urls.length > 0"
                class="modal-segmentation-mask-container"
              >
                <img
                  v-for="(maskUrl, maskIndex) in selectedImage.mask_urls"
                  :key="`modal-mask-${maskIndex}`"
                  :src="maskUrl"
                  class="modal-segmentation-mask"
                />
              </div>
            </div>
          </div>
          
          <div class="image-sidebar">
            <h3 class="sidebar-title">Image Details</h3>
            
            <div class="image-description">
              <h4>Description</h4>
              <p>{{ selectedImage.description || 'No description available' }}</p>
            </div>
            
            <div class="annotation-details">
              <h4>Annotations</h4>
              <div class="annotation-toggles">
                <div 
                  v-for="type in selectedAnnotationTypes"
                  :key="`toggle-${type}`"
                  class="annotation-toggle"
                  :class="{ active: zoomedAnnotations.includes(type) }"
                  @click="toggleZoomedAnnotationType(type)"
                >
                  <div class="toggle-switch">
                    <div class="toggle-indicator"></div>
                  </div>
                  <span class="toggle-label">{{ type }}</span>
                </div>
              </div>
              
              <!-- Details about annotations specific to this image -->
              <div v-if="activeAnnotations.includes('Object Classes')" class="object-classes">
                <h5>Object Classes</h5>
                <div class="object-tags">
                  <span v-for="(obj, idx) in selectedImage.objectClasses" :key="idx" class="object-tag">
                    {{ obj }}
                  </span>
                </div>
              </div>
              
              <div v-if="activeAnnotations.includes('Damage Type')" class="damage-types">
                <h5>Damage Types</h5>
                <div class="damage-tags">
                  <span v-for="(damage, idx) in selectedImage.damageTypes" :key="idx" class="damage-tag">
                    {{ damage }}
                  </span>
                </div>
              </div>
              
              <div v-if="activeAnnotations.includes('Severity Level')" class="severity-level">
                <h5>Severity Level</h5>
                <div class="severity-indicator" :class="selectedImage.severityLevel">
                  {{ selectedImage.severityLevel || 'N/A' }}
                </div>
              </div>
              
              <div v-if="activeAnnotations.includes('Descriptions')" class="descriptions">
                <h5>Detailed Descriptions</h5>
                <p class="detailed-description">{{ selectedImage.detailedDescription || 'No detailed description available' }}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script>
export default {
  data() {
    return {
      // Phase tracking
      selectionPhase: true,
      isZoomedImageReady: false,
      // Annotation types available
      annotationOptions: [
        "Bounding Boxes",
        "Segmentation",
        "Keypoints",
        "Descriptions",
        "Object Classes",
        "Damage Type",
        "Severity Level",
      ],
      activeAnnotations: [], // Global annotations
      zoomedAnnotations: [], // Annotations specific to the zoomed view
      // Selected annotation types (will not change after selection phase)
      selectedAnnotationTypes: [],
      // Loading state
      isLoading: false,
      loadingProgress: 0,
      // Mock data for annotated images
      annotatedImages: [],
      displayedImageDimensions: {},
      // Modal view state
      selectedImageIndex: null,
      zoomLevel: 100
    };
  },
  computed: {
    selectedImage() {
      return this.selectedImageIndex !== null ? this.annotatedImages[this.selectedImageIndex] : null;
    }
  },
  methods: {
    toggleZoomedAnnotationType(type) {
      if (this.zoomedAnnotations.includes(type)) {
        this.zoomedAnnotations = this.zoomedAnnotations.filter(t => t !== type);
      } else {
        this.zoomedAnnotations.push(type);
      }

      // Update masks if Segmentation was toggled
      if (type === 'Segmentation') {
        this.$nextTick(() => {
          if (this.selectedImage && this.selectedImage.mask_urls) {
            this.renderSegmentationMask(this.selectedImageIndex);
          }
        });
      }
    },
    
    calculateMaskStyle(index, maskIndex) {
      const dims = this.displayedImageDimensions[index];
      
      if (!dims) return {};
      
      // Generate a filter that will give this mask a unique color
      const hueRotation = (maskIndex * 60) % 360; // Different hue for each mask
      
      return {
        position: 'absolute',
        top: `${dims.top}px`,
        left: `${dims.left}px`,
        width: `${dims.width}px`,
        height: `${dims.height}px`,
        objectFit: 'contain',
        mixBlendMode: 'screen',
        opacity: 0.7,
        filter: `hue-rotate(${hueRotation}deg)`,
        pointerEvents: 'none',
      };
    },

    renderSegmentationMask(index) {
      const image = this.selectedImage;
      const dims = this.displayedImageDimensions[index];

      if (!dims || !image.masks || !image.shape) return;

      const canvas = document.getElementById(`mask-canvas-${index}`);
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      canvas.width = dims.width;
      canvas.height = dims.height;

      const scaleX = dims.width / image.shape[1];
      const scaleY = dims.height / image.shape[0];

      const imageData = ctx.createImageData(dims.width, dims.height);

      for (let y = 0; y < image.shape[0]; y++) {
        for (let x = 0; x < image.shape[1]; x++) {
          const displayX = Math.floor(x * scaleX);
          const displayY = Math.floor(y * scaleY);

          if (displayX >= 0 && displayX < dims.width && displayY >= 0 && displayY < dims.height) {
            const maskValue = image.masks[y][x];

            const idx = (displayY * dims.width + displayX) * 4;

            if (maskValue > 0) {
              imageData.data[idx] = 0;       // Red
              imageData.data[idx + 1] = 255; // Green
              imageData.data[idx + 2] = 0;   // Blue
              imageData.data[idx + 3] = 128; // Alpha (semi-transparent)
            } else {
              imageData.data[idx + 3] = 0; // Fully transparent for inactive pixels
            }
          }
        }
      }

      ctx.putImageData(imageData, 0, 0);
    },

    getColorForIndex(index) {
      // A list of distinct colors for visualization
      const colors = [
        'red', 'green', 'blue', 'orange', 'purple', 
        'cyan', 'magenta', 'yellow', 'lime', 'pink'
      ];
      return colors[index % colors.length];
    },
    
    calculateModalBoundingBoxStyle(bboxIndex) {
  const image = this.selectedImage;

  if (!image || !image.bboxes || !image.bboxes[bboxIndex] || !image.shape) {
    return {};
  }

  const bbox = image.bboxes[bboxIndex];
  const [x1, y1, x2, y2] = bbox;

  // Ensure the zoomedImageWrapper reference exists
  if (!this.$refs.zoomedImageWrapper) {
    console.warn("zoomedImageWrapper is not available yet.");
    return {};
  }

  // Get original image dimensions
  const originalWidth = image.shape[1];
  const originalHeight = image.shape[0];

  // Get displayed image dimensions
  const containerWidth = this.$refs.zoomedImageWrapper.offsetWidth;
  const containerHeight = this.$refs.zoomedImageWrapper.offsetHeight;

  // Calculate the aspect ratios
  const widthRatio = containerWidth / originalWidth;
  const heightRatio = containerHeight / originalHeight;

  // Use the smaller ratio to maintain aspect ratio (object-fit: contain behavior)
  const scaleFactor = Math.min(widthRatio, heightRatio);

  // Calculate the actual displayed image dimensions
  const displayedWidth = originalWidth * scaleFactor;
  const displayedHeight = originalHeight * scaleFactor;

  // Calculate the padding (empty space) due to object-fit: contain
  const leftPadding = (containerWidth - displayedWidth) / 2;
  const topPadding = (containerHeight - displayedHeight) / 2;

  // Scale the bounding box coordinates
  const left = leftPadding + x1 * scaleFactor;
  const top = topPadding + y1 * scaleFactor;
  const width = (x2 - x1) * scaleFactor;
  const height = (y2 - y1) * scaleFactor;

  const color = this.getColorForIndex(bboxIndex);

  return {
    position: 'absolute',
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`,
    border: `2px solid ${color}`,
    boxSizing: 'border-box',
    pointerEvents: 'none',
  };
},

    calculateModalMaskStyle(maskIndex) {
      return {
        position: 'absolute',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        objectFit: 'contain',
        pointerEvents: 'none',
      };
    },

    calculateBoundingBoxStyle(index, bboxIndex) {
      const image = this.annotatedImages[index];
      const dims = this.displayedImageDimensions[index];
      
      if (!dims || !image.bboxes || !image.bboxes[bboxIndex] || !image.shape) {
        return {};
      }
      
      const bbox = image.bboxes[bboxIndex];
      const [x1, y1, x2, y2] = bbox;
      
      // Get original image dimensions
      const originalWidth = image.shape[1];
      const originalHeight = image.shape[0];
      
      // Calculate the scaling factor
      const scaleX = dims.width / originalWidth;
      const scaleY = dims.height / originalHeight;
      
      // Calculate the position and size of the bounding box
      const left = dims.left + x1 * scaleX;
      const top = dims.top + y1 * scaleY;
      const width = (x2 - x1) * scaleX;
      const height = (y2 - y1) * scaleY;
      
      const color = this.getColorForIndex(bboxIndex);

      return {
        position: 'absolute',
        left: `${left}px`,
        top: `${top}px`,
        width: `${width}px`,
        height: `${height}px`,
        border: `2px solid ${color}`,
        boxSizing: 'border-box',
        pointerEvents: 'none',
      };
    },

    updateImageDimensions(event, index) {
      const imgElement = event.target;
      const containerWidth = imgElement.parentElement.offsetWidth;
      const containerHeight = imgElement.parentElement.offsetHeight;
      
      // Get the natural dimensions of the image
      const naturalWidth = imgElement.naturalWidth;
      const naturalHeight = imgElement.naturalHeight;
      
      // Calculate the actual dimensions of the displayed image with object-fit: contain
      let displayedWidth, displayedHeight;
      
      // Calculate scaling factors for both width and height
      const widthRatio = containerWidth / naturalWidth;
      const heightRatio = containerHeight / naturalHeight;
      
      // Use the smaller ratio to ensure the image fits completely
      const scaleFactor = Math.min(widthRatio, heightRatio);
      
      displayedWidth = naturalWidth * scaleFactor;
      displayedHeight = naturalHeight * scaleFactor;
      
      // Calculate the position (to account for centering of the contained image)
      const leftOffset = (containerWidth - displayedWidth) / 2;
      const topOffset = (containerHeight - displayedHeight) / 2;
      
      // Store all the calculated dimensions
      this.displayedImageDimensions[index] = {
        width: displayedWidth,
        height: displayedHeight,
        left: leftOffset,
        top: topOffset,
        containerWidth,
        containerHeight,
      };
      
      // Render segmentation mask if it exists and should be shown
      if (this.annotatedImages[index].masks && this.activeAnnotations.includes('Segmentation')) {
        // Wait for the next DOM update cycle to ensure the canvas elements are created
        this.$nextTick(() => {
          this.renderSegmentationMask(index);
        });
      }
    },

    toggleAnnotationSelection(option) {
      // This is used during the initial selection phase
      if (this.selectedAnnotationTypes.includes(option)) {
        this.selectedAnnotationTypes = this.selectedAnnotationTypes.filter(t => t !== option);
      } else {
        this.selectedAnnotationTypes.push(option);
      }
    },

    async proceedToLoading() {
      if (this.selectedAnnotationTypes.length === 0) return;

      // Lock in selections and proceed to loading
      this.selectionPhase = false;
      this.isLoading = true;

      // Initially, all selected types are active for viewing
      this.activeAnnotations = [...this.selectedAnnotationTypes];

      const userId = "unique_user_id"; // Replace with the actual user ID logic

      try {
        // Make API call to the backend
        const response = await fetch(`http://localhost:8000/annotate/?user_id=${userId}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            annotationTypes: this.selectedAnnotationTypes,
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to generate annotations");
        }

        // Process the streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;

          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            chunk.split("\n").forEach((line) => {
              if (line.trim()) {
                try {
                  const annotation = JSON.parse(line);
                  console.log("Received annotation:", annotation);

                  this.annotatedImages.push({
                    url: annotation.URL,
                    bboxes: annotation.bboxes,
                    mask_urls: annotation.mask_urls, // Use mask URL instead of RLE data
                    scores: annotation.scores,
                    shape: annotation.shape,
                  });
                } catch (error) {
                  console.error("Error parsing JSON:", error, "Line:", line);
                }
              }
            });
          }
        }

        console.log("Streaming complete");
        this.isLoading = false;
      } catch (error) {
        console.error("Error generating annotations:", error);
        this.isLoading = false;
        alert("Failed to generate annotations. Please try again.");
      }
    },
    simulateLoading() {
      // Simulate the loading of images with annotations
      const loadingInterval = setInterval(() => {
        this.loadingProgress += 5;
        if (this.loadingProgress >= 100) {
          clearInterval(loadingInterval);
          this.isLoading = false;
          this.generateMockImages();
        }
      }, 200);
    },
    generateMockImages() {
      // Generate mock data for demonstration
      this.annotatedImages = Array(12).fill().map((_, i) => ({
        id: i,
        url: `/api/placeholder/500/400`,
        description: `Sample annotated image ${i + 1}`,
        detailedDescription: `This is a detailed description of image ${i + 1} showing various objects with annotations applied based on the selected criteria.`,
        // Random annotations based on selected types
        hasBoundingBoxes: this.selectedAnnotationTypes.includes("Bounding Boxes"),
        hasSegmentation: this.selectedAnnotationTypes.includes("Segmentation"),
        hasKeypoints: this.selectedAnnotationTypes.includes("Keypoints"),
        objectClasses: ["Person", "Car", "Tree", "Building"].slice(0, Math.floor(Math.random() * 4) + 1),
        damageTypes: ["Scratch", "Dent", "Crack", "Tear"].slice(0, Math.floor(Math.random() * 3)),
        severityLevel: ["Low", "Medium", "High", "Critical"][Math.floor(Math.random() * 4)]
      }));
    },
    toggleAnnotationType(type) {
      // This is used after selection phase to toggle visibility
      if (this.activeAnnotations.includes(type)) {
        this.activeAnnotations = this.activeAnnotations.filter(t => t !== type);
      } else {
        this.activeAnnotations.push(type);
      }
      
      // Update masks if Segmentation was toggled
      if (type === 'Segmentation') {
        this.$nextTick(() => {
          this.annotatedImages.forEach((image, index) => {
            if (image.masks && this.displayedImageDimensions[index]) {
              this.renderSegmentationMask(index);
            }
          });
        });
      }
    },
    getAnnotationIcon(type) {
      // Return appropriate icon class based on annotation type
      const iconMap = {
        "Bounding Boxes": "icon-box",
        "Segmentation": "icon-segment",
        "Keypoints": "icon-point",
        "Descriptions": "icon-text",
        "Object Classes": "icon-tag",
        "Damage Type": "icon-damage",
        "Severity Level": "icon-severity"
      };
      return iconMap[type] || "icon-default";
    },
    getAnnotationDescription(type) {
      // Return description for each annotation type
      const descriptionMap = {
        "Bounding Boxes": "Rectangular borders around detected objects",
        "Segmentation": "Pixel-level masks highlighting object boundaries",
        "Keypoints": "Specific points marking features like joints or landmarks",
        "Descriptions": "Textual descriptions of image contents",
        "Object Classes": "Categories of objects identified in the image",
        "Damage Type": "Classification of damage patterns if present",
        "Severity Level": "Assessment of damage severity from low to critical"
      };
      return descriptionMap[type] || "";
    },

    getImageAnnotationCount(image) {
      // Count how many types of annotations this image has
      let count = 0;
      if (image.hasBoundingBoxes) count++;
      if (image.hasSegmentation) count++;
      if (image.hasKeypoints) count++;
      if (image.objectClasses && image.objectClasses.length) count++;
      if (image.damageTypes && image.damageTypes.length) count++;
      if (image.severityLevel) count++;
      if (image.detailedDescription) count++;
      return count;
    },

    selectImage(index) {
      this.selectedImageIndex = index;
      this.zoomLevel = 100; // Reset zoom when opening a new image

      // Disable all annotations by default in the zoomed view
      this.zoomedAnnotations = [];

      // Wait for the DOM to update before rendering bounding boxes
      this.$nextTick(() => {
        if (!this.$refs.zoomedImageWrapper) {
          console.warn("zoomedImageWrapper is still not available.");
          this.isZoomedImageReady = false;
        } else {
          console.log("zoomedImageWrapper is ready.");
          this.isZoomedImageReady = true; // Set the flag to true
        }
      });
    },

    closeImageModal() {
      this.selectedImageIndex = null;
    },
    changeZoom(amount) {
      this.zoomLevel = Math.max(50, Math.min(300, this.zoomLevel + amount));
    },
    resetZoom() {
      this.zoomLevel = 100;
    }
  }
};
</script>

<style scoped>
/* Global styles */
.annotation-viewer-app {
  background-color: #0f0f0f;
  color: #e0e0e0;
  min-height: 100vh;
  font-family: 'Inter', 'Roboto', sans-serif;
}

.bounding-box {
  position: absolute;
  border: 2px solid red;
  background-color: rgba(255, 0, 0, 0.2); /* Temporary background for debugging */
  box-sizing: border-box;
  pointer-events: none;
}

/* Header styles */
.app-header {
  background: linear-gradient(to right, #1e1e1e, #2d2d2d);
  padding: 1.5rem 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.app-title {
  font-size: 2rem;
  color: #00ffff;
  margin-bottom: 1rem;
  font-weight: 700;
}

.query-summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.query-card {
  background: #161616;
  border-left: 4px solid #00e676;
  border-radius: 8px;
  padding: 0.75rem 1.25rem;
  flex: 1;
  min-width: 250px;
}

.query-label {
  color: #00e676;
  font-weight: 600;
  font-size: 0.85rem;
  margin-right: 0.5rem;
}

.query-text {
  color: #ffffff;
  font-size: 1rem;
}

.annotations-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  max-width: 60%;
}

.annotation-tag {
  background: #00e676;
  color: #121212;
  padding: 0.5rem 1rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.2s ease;
  border: 1px solid rgba(0, 230, 118, 0.3);
}

/* Selection Phase Styles */
.selection-container {
  display: flex;
  justify-content: center;
  padding: 2rem;
}

.selection-content {
  width: 100%;
  max-width: 900px;
  background: #1e1e1e;
  border-radius: 20px;
  padding: 2.5rem;
  box-shadow: 0 0 30px rgba(0, 255, 255, 0.15);
}

.selection-heading {
  font-size: 1.8rem;
  color: #00ffff;
  margin-bottom: 1rem;
}

.selection-instruction {
  color: #aaaaaa;
  margin-bottom: 2rem;
  font-size: 1rem;
}

.annotation-options {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 2.5rem;
}

.annotation-option {
  display: flex;
  align-items: center;
  padding: 1rem;
  background: #121212;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid transparent;
  position: relative;
}

.annotation-option:hover {
  border-color: #00ffff;
}

.annotation-option.selected {
  background: rgba(0, 230, 118, 0.1);
  border-color: #00e676;
}

.option-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #2c2c2c;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 1rem;
}

.option-details {
  flex: 1;
}

.option-name {
  display: block;
  font-weight: 600;
  margin-bottom: 0.25rem;
  font-size: 1.1rem;
}

.option-description {
  display: block;
  color: #888888;
  font-size: 0.9rem;
}

.selection-indicator {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 2px solid #555555;
  transition: all 0.2s ease;
}

.annotation-option.selected .selection-indicator {
  background: #00e676;
  border-color: #00e676;
  box-shadow: 0 0 10px rgba(0, 230, 118, 0.5);
}

.selection-actions {
  display: flex;
  justify-content: center;
}

.proceed-button {
  background: linear-gradient(to right, #00e676, #00ffff);
  color: #121212;
  font-weight: 700;
  font-size: 1.1rem;
  padding: 1rem 2.5rem;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 5px 15px rgba(0, 230, 118, 0.4);
}

.proceed-button:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(0, 230, 118, 0.5);
}

.proceed-button:disabled {
  background: #333333;
  color: #777777;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

/* Icon classes for annotation types */
.icon-box {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="%2300e676" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" stroke="%2300e676" fill="none" stroke-width="2"/></svg>');
}

.icon-segment {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="%2300e676" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="%2300e67633" stroke="%2300e676" stroke-width="2"/></svg>');
}

.icon-point {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="%2300e676" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3" fill="%2300e676"/><circle cx="6" cy="6" r="3" fill="%2300e676"/><circle cx="18" cy="18" r="3" fill="%2300e676"/></svg>');
}

.icon-text {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="%2300e676" viewBox="0 0 24 24"><path d="M3 5h18v2H3V5zm0 6h18v2H3v-2zm0 6h18v2H3v-2z" fill="%2300e676"/></svg>');
}

.icon-tag {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="%2300e676" viewBox="0 0 24 24"><path d="M21.41 11.58l-9-9C12.04 2.21 11.53 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .53.21 1.04.59 1.41l9 9c.78.78 2.05.78 2.83 0l7-7c.78-.78.78-2.05-.01-2.83zM6.5 8C5.67 8 5 7.33 5 6.5S5.67 5 6.5 5 8 5.67 8 6.5 7.33 8 6.5 8z" fill="%2300e676"/></svg>');
}

.icon-damage {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="%2300e676" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="%2300e676"/></svg>');
}

.icon-severity {
  background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" fill="%2300e676" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" fill="%2300e676"/></svg>');
}

/* Loading state */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 70vh;
  padding: 2rem;
}

.loading-spinner {
  width: 60px;
  height: 60px;
  border: 5px solid #00ffff33;
  border-top: 5px solid #00ffff;
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
  margin-bottom: 1.5rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  font-size: 1.2rem;
  margin-bottom: 2rem;
  color: #00e676;
}

.progress-bar {
  width: 80%;
  max-width: 500px;
  height: 8px;
  background: #2c2c2c;
  border-radius: 999px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(to right, #00e676, #00ffff);
  border-radius: inherit;
  transition: width 0.3s ease;
}

/* Filter bar */
.annotation-filter-bar {
  position: sticky;
  top: 0;
  background: rgba(30, 30, 30, 0.95);
  backdrop-filter: blur(10px);
  padding: 1rem 2rem;
  z-index: 10;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 2px 15px rgba(0, 0, 0, 0.2);
}

.filter-title {
  font-weight: 600;
  color: #00ffff;
  font-size: 0.95rem;
}

.filter-options {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.filter-chip {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  background: #272727;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid transparent;
}

.filter-chip:hover {
  border-color: #00ffff;
}

.filter-chip.active {
  background: #00e676;
  color: #121212;
  font-weight: 600;
}

.filter-icon {
  width: 16px;
  height: 16px;
  display: inline-block;
}

.filter-chip.active .filter-icon {
  filter: brightness(0); /* Change icon color to black when active */
}

/* Image gallery grid */
.image-gallery-container {
  padding: 1.5rem;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
  padding: 1.5rem 0;
}

.image-card {
  background: #1a1a1a;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.image-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.image-container {
  position: relative;
  aspect-ratio: 5/4;
  overflow: hidden;
}

.image-thumbnail {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.image-metadata {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
  padding: 1rem;
  color: white;
}

.image-count {
  background: rgba(0, 230, 118, 0.8);
  color: #121212;
  font-size: 0.75rem;
  padding: 0.3rem 0.6rem;
  border-radius: 999px;
  font-weight: 600;
}

/* Annotation overlays */
.annotation-overlays {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.annotation-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

/* Placeholder for annotation visualizations */
.bounding-box-placeholder {
  /* Simulation of random bounding boxes */
  border: 2px solid rgba(255, 204, 0, 0.7);
  position: absolute;
  width: 40%;
  height: 60%;
  top: 20%;
  left: 30%;
}

.segmentation-placeholder {
  /* Simulation of segmentation mask */
  position: absolute;
  width: 60%;
  height: 40%;
  top: 30%;
  left: 20%;
  background: rgba(255, 0, 128, 0.3);
  border-radius: 50%;
}

.keypoints-placeholder {
  /* Simulation of keypoints */
  position: absolute;
  width: 100%;
  height: 100%;
}

.keypoints-placeholder::before,
.keypoints-placeholder::after {
  content: '';
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #00ffff;
  top: 40%;
  left: 40%;
}

.keypoints-placeholder::after {
  top: 60%;
  left: 60%;
}

/* Modal for zoomed image view */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  padding: 2rem;
}

.modal-content {
  display: flex;
  flex-direction: row; /* Ensure the image and sidebar are side by side */
  height: 90vh; /* Fixed height for the modal */
  max-height: 90vh;
  overflow: hidden;
}

.close-modal-btn {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: rgba(255, 255, 255, 0.1);
  color: #ffffff;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  border: none;
  font-size: 1.5rem;
  line-height: 1;
  cursor: pointer;
  z-index: 10;
  transition: background 0.2s;
}

.close-modal-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.modal-image-container {
  flex: 2; /* Take up 2/3 of the modal width */
  position: relative;
  background: #0a0a0a;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%; /* Ensure the image container takes the full height of the modal */
}

.zoom-controls {
  position: absolute;
  bottom: 1rem;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.6);
  padding: 0.5rem;
  border-radius: 999px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  z-index: 5;
}

.zoom-btn {
  background: rgba(255, 255, 255, 0.15);
  color: white;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.zoom-btn.reset {
  width: auto;
  padding: 0 1rem;
  border-radius: 999px;
  font-size: 0.85rem;
}

.zoom-level {
  color: white;
  font-size: 0.9rem;
  min-width: 45px;
  text-align: center;
}

.zoomed-image-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  transform-origin: center; /* Ensure scaling happens from the center */
  transition: transform 0.2s ease; /* Smooth scaling */
  background-color: #000; /* Black background for the zoomed view */
  display: flex; /* Ensure the image is centered */
  align-items: center;
  justify-content: center;
}

.zoomed-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  z-index: 1; /* Ensure the image is below the bounding boxes and masks */
}

/* Modal annotation layers */
.modal-annotation-layer {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  transform-origin: center; /* Ensure annotations scale from the center */
  transition: transform 0.2s ease;
}

.modal-bounding-box {
  position: absolute;
  border: 2px solid red;
  box-sizing: border-box;
  pointer-events: none;
}

.modal-segmentation {
  /* More detailed segmentation simulation for modal */
  position: absolute;
  width: 60%;
  height: 40%;
  top: 30%;
  left: 20%;
  background: rgba(255, 0, 128, 0.4);
  border-radius: 50%;
  box-shadow: 0 0 20px rgba(255, 0, 128, 0.3);
}

.modal-keypoints {
  /* More detailed keypoints simulation for modal */
  position: absolute;
  width: 100%;
  height: 100%;
}

.modal-keypoints::before,
.modal-keypoints::after {
  content: '';
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #00ffff;
  box-shadow: 0 0 10px rgba(0, 255, 255, 0.7);
  top: 40%;
  left: 40%;
}

.modal-keypoints::after {
  top: 60%;
  left: 60%;
}

/* Image sidebar */
.image-sidebar {
  flex: 1; /* Take up 1/3 of the modal width */
  background: #0d0d0d;
  padding: 1.5rem;
  overflow-y: auto; /* Make the sidebar scrollable if content overflows */
  height: 100%; /* Ensure the sidebar takes the full height of the modal */
  border-left: 1px solid rgba(255, 255, 255, 0.1);
}

.sidebar-title {
  color: #00ffff;
  margin-bottom: 1.5rem;
  font-size: 1.3rem;
}

.image-description {
  margin-bottom: 2rem;
}

.image-description h4 {
  color: #00e676;
  margin-bottom: 0.5rem;
}

.annotation-details h4 {
  color: #00e676;
  margin-bottom: 1rem;
}

.annotation-toggles {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 2rem;
}

.annotation-toggle {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
}

.toggle-switch {
  width: 40px;
  height: 20px;
  background: #333;
  border-radius: 999px;
  position: relative;
  transition: background 0.2s;
}

.toggle-indicator {
  width: 16px;
  height: 16px;
  background: #fff;
  border-radius: 50%;
  position: absolute;
  top: 2px;
  left: 2px;
  transition: transform 0.2s;
}

.annotation-toggle.active .toggle-switch {
  background: #00e676;
}

.annotation-toggle.active .toggle-indicator {
  transform: translateX(20px);
}

/* Object classes, damage types, etc. */
.object-classes,
.damage-types,
.severity-level,
.descriptions {
  margin-bottom: 1.5rem;
}

.object-classes h5,
.damage-types h5,
.severity-level h5,
.descriptions h5 {
  color: #ffffff;
  font-size: 0.9rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
}

.object-tags,
.damage-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.object-tag {
  background: rgba(0, 204, 255, 0.2);
  border: 1px solid rgba(0, 204, 255, 0.4);
  color: #00ccff;
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  font-size: 0.8rem;
}

.damage-tag {
  background: rgba(255, 102, 0, 0.2);
  border: 1px solid rgba(255, 102, 0, 0.4);
  color: #ff6600;
  padding: 0.3rem 0.6rem;
  border-radius: 4px;
  font-size: 0.8rem;
}

.severity-indicator {
  display: inline-block;
  padding: 0.4rem 0.8rem;
  border-radius: 4px;
  font-size: 0.85rem;
  font-weight: 600;
}

.severity-indicator.Low {
  background: rgba(0, 204, 102, 0.2);
  color: #00cc66;
}

.severity-indicator.Medium {
  background: rgba(255, 204, 0, 0.2);
  color: #ffcc00;
}

.severity-indicator.High {
  background: rgba(255, 153, 0, 0.2);
  color: #ff9900;
}

.severity-indicator.Critical {
  background: rgba(255, 51, 51, 0.2);
  color: #ff3333;
}

.detailed-description {
  line-height: 1.6;
  color: #cccccc;
  font-size: 0.9rem;
}

/* Animation classes */
.modal-enter-active, .modal-leave-active {
  transition: opacity 0.3s;
}
.modal-enter, .modal-leave-to {
  opacity: 0;
}

/* Media queries for responsiveness */
@media (max-width: 1100px) {
  .modal-content {
    flex-direction: column;
    height: auto;
    max-height: 90vh;
  }
  
  .image-sidebar {
    width: 100%;
    max-height: 40vh;
    border-left: none;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }
}

@media (max-width: 768px) {
  .query-summary {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .annotations-summary {
    max-width: 100%;
  }
  
  .annotation-filter-bar {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .image-grid {
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  }
}

@media (max-width: 480px) {
  .app-header {
    padding: 1rem;
  }
  
  .image-grid {
    grid-template-columns: 1fr;
  }
}

.segmentation-mask-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Allow clicking through the mask */
}

.segmentation-mask {
  position: absolute;
  pointer-events: none;
  filter: hue-rotate(120deg); /* Makes white areas green */
  max-width: 100%;
  max-height: 100%;
  transform: translateX(0); /* Reset any transform */
}

.modal-segmentation-mask {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain; /* Ensure the mask scales with the image */
  opacity: 0.7; /* Semi-transparent mask */
  filter: hue-rotate(120deg);
  pointer-events: none;
}

.bounding-boxes-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.modal-bounding-boxes-container,
.modal-segmentation-mask-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none; /* Allow clicking through the mask */
  z-index: 2; /* Ensure they are above the image */
}

</style>
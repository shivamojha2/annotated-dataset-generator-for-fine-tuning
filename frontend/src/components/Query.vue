<template>
  <div class="query-app">
    <header class="app-header">
      <h1 class="app-title">AI Visual Query</h1>
    </header>

    <!-- Query input section -->
    <div class="query-container">
      <div class="query-input-wrapper">
        <label for="query-input" class="query-label">Describe your image query</label>
        <input
          id="query-input"
          v-model="query"
          type="text"
          class="query-input"
          placeholder="e.g., Find car images with windshield damage"
          @keyup.enter="fetchData"
        />
        <button class="query-button" @click="fetchData" :disabled="!query.trim()">Search</button>
      </div>
      <p class="query-example">Example: "Find car images with windshield damage"</p>
    </div>

    <!-- Loading state -->
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p class="loading-text">Fetching images...</p>
    </div>

    <!-- Images container -->
    <div v-if="images.length && !loading" class="image-gallery-container">
      <div class="image-gallery-wrapper">
        <h2 class="section-heading">Select Relevant Images</h2>
        <div class="image-grid">
          <div
            v-for="(img, index) in images"
            :key="index"
            class="image-card"
            :class="{ selected: selectedImages.includes(img) }"
            @click="toggleSelection(img)"
          >
            <img :src="img" alt="Image" class="image-thumbnail" />
          </div>
        </div>
        <button class="submit-button" @click="submitSelection" :disabled="!selectedImages.length">
          Submit Selection
        </button>
      </div>
    </div>

    <!-- Final response -->
    <div v-if="finalResponse" class="response-message">
      {{ finalResponse }}
    </div>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      query: "",
      loading: false,
      images: [],
      selectedImages: [],
      finalResponse: null,
    };
  },
  methods: {
    async fetchData() {
      if (!this.query.trim()) {
        this.finalResponse = "Please enter a valid query.";
        return;
      }

      this.loading = true;
      this.finalResponse = null;
      try {
        const response = await axios.post("http://localhost:8000/query/", {
          query: this.query,
        });

        this.images = response.data.images;
        this.selectedImages = [];
      } catch (error) {
        this.finalResponse =
          "Error fetching images: " +
          (error.response?.data?.detail || error.message);
      } finally {
        this.loading = false;
      }
    },
    toggleSelection(image) {
      if (this.selectedImages.includes(image)) {
        this.selectedImages = this.selectedImages.filter((img) => img !== image);
      } else {
        this.selectedImages.push(image);
      }
    },
    async submitSelection() {
      if (!this.selectedImages.length) {
        this.finalResponse = "Please select at least one image.";
        return;
      }

      this.finalResponse = "Processing your selections...";
      this.loading = true;

      try {
        const response = await axios.post("http://localhost:8000/query/", {
          selected_images: this.selectedImages,
        });
        this.finalResponse = response.data.message || "Selection submitted!";

        // Keep loading state true and navigate immediately
        setTimeout(() => {
          this.$router.push({ name: "annotate", query: { query: this.query } });
        }, 500); // Navigate after a short delay
      } catch (error) {
        console.error("Error submitting selection:", error);
        this.finalResponse = "Something went wrong. Please try again.";
        this.loading = false; // Allow retry if there's an error
      }
    },
  },
};
</script>

<style scoped>
.query-app {
  background-color: #0f0f0f;
  color: #e0e0e0;
  min-height: 100vh;
  font-family: 'Inter', 'Roboto', sans-serif;
  padding: 2rem;
}

.app-header {
  text-align: center;
  margin-bottom: 2rem;
}

.app-title {
  font-size: 2.5rem;
  color: #00ffff;
  font-weight: 700;
}

.query-container {
  max-width: 600px;
  margin: 0 auto 2rem;
  text-align: center;
}

.query-input-wrapper {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.query-label {
  font-size: 1rem;
  color: #aaaaaa;
}

.query-input {
  padding: 1rem;
  border: none;
  border-radius: 12px;
  background: #1e1e1e;
  color: #ffffff;
  font-size: 1rem;
  outline: none;
}

.query-input::placeholder {
  color: #555555;
}

.query-button {
  background: linear-gradient(to right, #00e676, #00ffff);
  color: #121212;
  font-weight: 700;
  font-size: 1rem;
  padding: 0.75rem 2rem;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 5px 15px rgba(0, 230, 118, 0.4); /* Add shadow for consistency */
}

.query-button:hover {
  transform: translateY(-3px); /* Lift the button slightly on hover */
  box-shadow: 0 8px 20px rgba(0, 230, 118, 0.5); /* Enhance shadow on hover */
}

.query-button:disabled {
  background: #333333;
  color: #777777;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.query-example {
  font-size: 0.85rem;
  color: #888888;
  margin-top: 1rem;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 50vh;
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #00ffff33;
  border-top: 5px solid #00ffff;
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
  margin-bottom: 1rem;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.loading-text {
  font-size: 1.2rem;
  color: #00e676;
}

.image-gallery-container {
  padding: 2rem 0;
}

.image-gallery-container.hidden {
  opacity: 0; /* Hide content smoothly */
}

.image-gallery-wrapper {
  max-width: 900px;
  margin: 0 auto;
  text-align: center;
}

.section-heading {
  font-size: 1.8rem;
  color: #00ffff;
  margin-bottom: 1.5rem;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr); /* 3 images per row */
  gap: 1.5rem; /* Spacing between images */
  justify-items: center; /* Center images horizontally */
}

.image-card {
  position: relative;
  width: 100%; /* Ensure the card takes full width of the grid cell */
  max-width: 250px; /* Limit the card width */
  border-radius: 12px;
  overflow: hidden;
  cursor: pointer;
  background: #1a1a1a;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Subtle shadow for the card */
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.image-card:hover {
  transform: translateY(-5px); /* Lift the card slightly on hover */
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3); /* Enhance shadow on hover */
}

.image-card.selected {
  border: 2px solid #00e676; /* Highlight selected images */
}

.image-thumbnail {
  width: 100%; /* Ensure the image fills the card's width */
  height: 200px; /* Fixed height for consistent layout */
  object-fit: contain; /* Show the full image without cropping */
  border-radius: 12px; /* Keep the rounded corners */
  background-color: #232121; /* Add a background color for empty space */
}

.submit-button {
  display: block;
  margin: 2rem auto 0;
  background: linear-gradient(to right, #00e676, #00ffff);
  color: #121212;
  font-weight: 700;
  font-size: 1rem;
  padding: 0.75rem 2rem;
  border: none;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 5px 15px rgba(0, 230, 118, 0.4); /* Add shadow for consistency */
}

.submit-button:hover {
  transform: translateY(-3px); /* Lift the button slightly on hover */
  box-shadow: 0 8px 20px rgba(0, 230, 118, 0.5); /* Enhance shadow on hover */
}

.submit-button:disabled {
  background: #333333;
  color: #777777;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.response-message {
  text-align: center;
  margin-top: 2rem;
  padding: 1rem;
  background: #1b5e20;
  border-radius: 12px;
  color: #ffffff;
  font-size: 1rem;
}
</style>
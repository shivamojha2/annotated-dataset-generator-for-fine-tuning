<template>
  <RouterView />
</template>

<script setup>
import { RouterView } from 'vue-router'
</script>

<style>
html, body {
  background-color: #181818 !important;
  color: #e0e0e0 !important;
}

#app {
  width: 100%;
  height: 100%;
  margin-top: 60px;
  overflow: hidden;
}
</style>

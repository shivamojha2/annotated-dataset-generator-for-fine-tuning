import { createRouter, createWebHistory } from 'vue-router'
import Query from '../components/Query.vue'
import Annotate from '../components/Annotate.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/query',
      name: 'query',
      component: Query,
    },
    {
      path: '/annotate',
      name: 'annotate',
      component: Annotate,
    },
  ]
})

export default router
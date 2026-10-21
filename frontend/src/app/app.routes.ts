import { Routes } from '@angular/router';

const APP = 'Meeting Intelligence';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
  {
    path: 'dashboard', title: `Dashboard · ${APP}`,
    loadComponent: () => import('./components/dashboard/dashboard.component').then((m) => m.DashboardComponent),
  },
  {
    path: 'meetings', title: `Meetings · ${APP}`,
    loadComponent: () => import('./components/meeting-list/meeting-list.component').then((m) => m.MeetingListComponent),
  },
  {
    path: 'meetings/:id', title: `Meeting · ${APP}`,
    loadComponent: () => import('./components/meeting-detail/meeting-detail.component').then((m) => m.MeetingDetailComponent),
  },
  {
    path: 'meetings/:id/progress', title: `Processing · ${APP}`,
    loadComponent: () => import('./components/processing-status/processing-status.component').then((m) => m.ProcessingStatusComponent),
  },
  { path: 'processing/:id', redirectTo: 'meetings/:id/progress' },
  {
    path: 'upload', title: `Upload recording · ${APP}`,
    loadComponent: () => import('./components/upload/upload.component').then((m) => m.UploadComponent),
  },
  {
    path: 'search', title: `Search · ${APP}`,
    loadComponent: () => import('./components/search/search.component').then((m) => m.SearchComponent),
  },
  {
    // Administrators only (the API requires the admin key); deliberately not in the navigation.
    path: 'admin/settings', title: `Administration · ${APP}`,
    loadComponent: () => import('./components/settings/settings.component').then((m) => m.SettingsComponent),
  },
  { path: 'settings', redirectTo: 'admin/settings' },
  {
    path: '**', title: `Page not found · ${APP}`,
    loadComponent: () => import('./components/not-found/not-found.component').then((m) => m.NotFoundComponent),
  },
];

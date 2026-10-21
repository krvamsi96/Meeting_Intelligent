// Development build. The API is reached through the dev-server proxy (proxy.conf.json -> the backend on port 8000),
// so no backend address is compiled into the app.
export const environment = {
  production: false,
  apiUrl: '/api/v1',
};

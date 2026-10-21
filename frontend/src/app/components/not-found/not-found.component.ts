import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-not-found',
  imports: [RouterLink],
  template: `
    <div class="page page-narrow">
      <div class="card state">
        <h1 class="page-title">Page not found</h1>
        <p>The page you are looking for does not exist or has moved.</p>
        <a routerLink="/dashboard" class="btn btn-primary">Go to the dashboard</a>
      </div>
    </div>`,
})
export class NotFoundComponent {}

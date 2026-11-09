import { Routes } from '@angular/router';
import { Overview } from './components/overview/overview';
import { Recording } from './components/recording/recording';

export const routes: Routes = [
    {
        path: 'overview',
        component: Overview
    },
    {
        path: 'recording',
        component: Recording
    },
    {
        path: '',
        redirectTo: '/recording',
        pathMatch: 'full'
    }
];

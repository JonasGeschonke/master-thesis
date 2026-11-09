import { Component } from '@angular/core';
import { LucideAngularModule, Mic, Library, BarChart3 } from 'lucide-angular';

@Component({
  selector: 'app-navbar',
  imports: [LucideAngularModule],
  templateUrl: './navbar.html',
  styleUrl: './navbar.css'
})
export class Navbar {
  readonly iconMic = Mic;
  readonly iconLibrary = Library;
  readonly iconChart = BarChart3;
}

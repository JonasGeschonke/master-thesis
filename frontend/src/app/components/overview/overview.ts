import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { FormControl, ReactiveFormsModule, Validators } from '@angular/forms';
import { BackendService } from '../../service/backend';

@Component({
  selector: 'app-overview',
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './overview.html',
  styleUrl: './overview.css'
})
export class Overview {
  backendService = inject(BackendService);
  textControl = new FormControl<string>('', [Validators.required, Validators.minLength(5)]);

  submitText() {
    if (this.textControl.valid) {
      console.log(this.textControl.value);
      this.backendService.saveText(this.textControl.value!).subscribe(
        (response) => {
          console.log(response);
        }
      );
    }
  }
}

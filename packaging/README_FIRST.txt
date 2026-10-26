مدرّب العين الرقمية - Digital Eye Trainer
==========================================

التشغيل:
  انقر مرتين على DigitalEyeTrainer.exe
  تُفتح صفحة في المتصفح (http://127.0.0.1:8765) تعمل على جهازك فقط.
  لا يحتاج إنترنت ولا تثبيت بايثون: النماذج الأساسية موجودة داخل البرنامج.
  لإيقاف البرنامج أغلق النافذة السوداء.

أين تُحفظ بياناتك؟
  في مجلد "workspace" بجانب البرنامج (وثائقك، التصحيحات، النماذج، التقارير).
  ضع مجلد البرنامج في مكان عادي مثل C:\DigitalEyeTrainer
  وليس داخل OneDrive / Google Drive / Dropbox (هذه ترفع الملفات للإنترنت).

ماذا أشارك؟
  فقط workspace\reports (أرقام فقط). لا تشارك workspace\private ولا workspace\exports:
  النموذج المدرَّب سرّي لأنه قد يحفظ أسماء وأرقاماً من وثائقك.

إذا ظهر تحذير "Windows protected your PC":
  اضغط More info ثم Run anyway. يظهر لأن البرنامج غير موقّع رقمياً.

----------------------------------------------------------------------
Start: double-click DigitalEyeTrainer.exe. A page opens on http://127.0.0.1:8765
(this computer only). No internet and no Python needed. Close the black window to stop.
Your data is in the "workspace" folder next to the program. Do not put this folder in a
cloud-synced folder. Share only workspace\reports. The exported model is confidential.
Command line: DigitalEyeTrainer.exe check | import FOLDER | train | evaluate | export | demo

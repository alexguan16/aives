-keep class org.beyka.tiffbitmapfactory.**{ *; }
-keep class org.mp4parser.**{ *; }
-keep class ai.onnxruntime.** { *; }

# referenced from: com.google.crypto.tink
-dontwarn com.google.errorprone.annotations.**
-dontwarn javax.annotation.**

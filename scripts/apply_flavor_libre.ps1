$PUBSPEC_PATH = "pubspec.yaml"

if ((split-path -path (get-location) -leaf) -eq "scripts") {
	cd ..
}

# no flutterw for windows
flutter clean

(get-content $PUBSPEC_PATH) | %{$_ -replace "plugins/aves_services_.*","plugins/aves_services_none"} | set-content $PUBSPEC_PATH
(get-content $PUBSPEC_PATH) | %{$_ -replace "plugins/aves_report_.*","plugins/aves_report_console"} | set-content $PUBSPEC_PATH

flutter pub get
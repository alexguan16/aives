import 'package:aves/model/settings/defaults.dart';
import 'package:aves_model/aves_model.dart';

mixin MLSettings on SettingsAccess {
  bool get mlCustom => getBool(SettingKeys.mlCustom) ?? SettingsDefaults.mlCustom;

  String? get mlConfig => getString(SettingKeys.mlConfig);
}
#include "camera_commands.hpp"

CommandResult updateCameraCommand(std::shared_ptr<DependencyRegistry> registry, const nlohmann::json& json)
{
    auto payload = json.get<UpdateCameraConfigPayload>();

    std::shared_ptr<ProjectConfig> projectConfig = registry->resolve<ProjectConfig>(DependencyType::project_config);
    auto oldConfig = projectConfig->getCameraConfig();
    projectConfig->setCameraConfig(
        payload.vflip.has_value() ? payload.vflip.value() : oldConfig.vflip, payload.framesize.has_value() ? payload.framesize.value() : oldConfig.framesize,
        payload.href.has_value() ? payload.href.value() : oldConfig.href, payload.quality.has_value() ? payload.quality.value() : oldConfig.quality,
        payload.brightness.has_value() ? payload.brightness.value() : oldConfig.brightness);

  return CommandResult::getSuccessResult("Config updated");
}

CommandResult setEncodingModeCommand(std::shared_ptr<DependencyRegistry> registry, const nlohmann::json &json)
{
  auto payload = json.get<SetEncodingModePayload>();

  if (payload.mode > 1) {
    return CommandResult::getErrorResult("Invalid encoding mode (0=JPEG, 1=JPEGLS)");
  }

  std::shared_ptr<ProjectConfig> projectConfig = registry->resolve<ProjectConfig>(DependencyType::project_config);
  auto mode = static_cast<EncodingMode>(payload.mode);
  projectConfig->setEncodingMode(mode);

  std::shared_ptr<CameraManager> cameraManager = registry->resolve<CameraManager>(DependencyType::camera_manager);
  cameraManager->setPixelFormat(mode == EncodingMode::JPEGLS ? PIXFORMAT_GRAYSCALE : PIXFORMAT_JPEG);

  return CommandResult::getSuccessResult(mode == EncodingMode::JPEGLS ? "Switched to JPEG-LS" : "Switched to JPEG");
}

CommandResult getEncodingModeCommand(std::shared_ptr<DependencyRegistry> registry)
{
  std::shared_ptr<ProjectConfig> projectConfig = registry->resolve<ProjectConfig>(DependencyType::project_config);
  auto mode = projectConfig->getEncodingMode();
  nlohmann::json result;
  result["encoding_mode"] = static_cast<int>(mode);
  result["encoding_mode_name"] = (mode == EncodingMode::JPEGLS) ? "jpegls" : "jpeg";
  return CommandResult::getSuccessResult(result);
}
#ifndef CAMERA_COMMANDS_HPP
#define CAMERA_COMMANDS_HPP
#include <CameraManager.hpp>
#include <ProjectConfig.hpp>
#include <memory>
#include <nlohmann-json.hpp>
#include <optional>
#include <string>
#include "CommandResult.hpp"
#include "CommandSchema.hpp"
#include "DependencyRegistry.hpp"

CommandResult updateCameraCommand(std::shared_ptr<DependencyRegistry> registry, const nlohmann::json &json);
CommandResult setEncodingModeCommand(std::shared_ptr<DependencyRegistry> registry, const nlohmann::json &json);
CommandResult getEncodingModeCommand(std::shared_ptr<DependencyRegistry> registry);

#endif

// add cropping command
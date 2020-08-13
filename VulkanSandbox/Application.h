#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//#include <iostream>
//#include <stdexcept>
#include <functional>
//#include <cstdlib>
#include <optional>

class GLFWwindow;

class Application
{
public:
    void run();
private:
    GLFWwindow* m_window = nullptr;

    VkInstance m_instance;
	
    VkSurfaceKHR m_surface;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device;
	
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
	
    VkSwapchainKHR m_swapChain;
    std::vector<VkImage> m_swapChainImages;
    VkFormat m_swapChainImageFormat;
    VkExtent2D m_swapChainExtent;

    std::vector<VkImageView> m_swapChainImageViews;

    VkRenderPass m_renderPass;
    VkPipelineLayout m_pipelineLayout;

    VkPipeline m_graphicsPipeline;

    std::vector<VkFramebuffer> m_swapChainFramebuffers;

    VkCommandPool m_commandPool;
    std::vector<VkCommandBuffer> m_commandBuffers;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    std::vector<VkFence> m_imagesInFlight;
    size_t m_currentFrame = 0;
#ifdef NDEBUG
    static const bool m_enableValidationLayers = false;
#else
    static const bool m_enableValidationLayers = true;
#endif
	
    static const std::vector<const char*> m_validationLayers;
    static const std::vector<const char*> m_deviceExtensions;
	
    struct QueueFamilyIndices
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() const
        {
            return graphicsFamily.has_value() &&
                presentFamily.has_value();
        }
    };

	struct SwapChainSupportDetails
	{
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
	};
private:
    void initWindow();

    void initVulkan();
    void createInstance();
	
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();

    void createSwapChain();

    void createImageViews();

    void createRenderPass();

    void createGraphicsPipeline();

    void createFramebuffers();

    void createCommandPool();
    void createCommandBuffers();

    void drawFrame();

    void createSyncObjects();
	
    void mainLoop();
    void cleanup();

	// extension helper functions
    bool validateExtensions(const char** extensionsToValidate, uint32_t count) const;
    uint32_t getExtensionCount() const;

	// validation layer helper functions
    bool checkValidationLayerSupport(const char* const* layersToValidate, uint32_t count) const;
    uint32_t getAvailableValidationLayerCount() const;

	// physical device and queue family helper functions
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

	// swap chain helper functions
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    static VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
	
	// graphics pipeline helper functions
    VkShaderModule createShaderModule(const std::vector<char>& code);
};

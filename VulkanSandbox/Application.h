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

    void framebufferResized();
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
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;

    VkPipeline m_graphicsPipeline;

    std::vector<VkFramebuffer> m_swapChainFramebuffers;

    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;
    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformBuffersMemory;
	
    VkImage m_textureImage;
    VkDeviceMemory m_textureImageMemory;

    VkCommandPool m_commandPool;
    std::vector<VkCommandBuffer> m_commandBuffers;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    std::vector<VkFence> m_imagesInFlight;
    size_t m_currentFrame = 0;

    bool m_framebufferResized = false;
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
    void cleanupSwapChain();
    void recreateSwapChain();

    void createImageViews();

    void createRenderPass();

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    
    void createGraphicsPipeline();

    void createFramebuffers();

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
	
    void createCommandPool();
    void createCommandBuffers();

    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    void createImage(uint32_t width, uint32_t height, VkFormat format,
        VkImageTiling tiling, VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties, VkImage& image,
        VkDeviceMemory& imageMemory);
    void createTextureImage();
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

    void drawFrame();
    void updateUniformBuffer(uint32_t currentImage);

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
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	// swap chain helper functions
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
	
	// graphics pipeline helper functions
    VkShaderModule createShaderModule(const std::vector<char>& code);
};


// TRX-GPU Unleashed - pomocnicze funkcje Vulkan (Linux / Windows / macOS przez MoltenVK).
//
// Co robi:
//   1. Wybiera rodziny kolejek: graphics + dedykowane compute / transfer (praca asynchroniczna).
//   2. Wykrywa ReBAR / SAM / UMA i alokuje bufory z fallbackami (bezposredni zapis do VRAM,
//      gdy sie oplaca; staging, gdy nie).
//   3. Pilnuje budzetu pamieci (VK_EXT_memory_budget), zeby nie doprowadzic do pagingu / TDR.
//   4. Semafory timeline + Submit bez alokacji na stercie.
//   5. Pipeline cache na dysku z walidacja naglowka i atomowym zapisem.
//   6. Obsluga MoltenVK (portability enumeration / portability subset).
//
// Czego NIE robi: nie omija zadnych zabezpieczen, nie siega do sprzetu poza Vulkanem.
// Wymagania: Vulkan 1.2+ (instancja tez), C++20, linkowanie z loaderem Vulkana.
// Alokator tworzy osobna alokacje na bufor (limit maxMemoryAllocationCount ~4096) - przy
// tysiacach malych buforow uzyj sub-alokacji (np. VMA) albo istniejacego alokatora Kyty.
//
// Przyklad:
//   auto fam = TrxGpu::PickQueueFamilies(gpu);
//   auto sup = TrxGpu::QueryDeviceSupport(gpu);      // sup.extensions -> VkDeviceCreateInfo
//   // ... utworz VkDevice (wlacz timelineSemaphore, jesli sup.timelineSemaphore) ...
//   TrxGpu::Allocator alloc(gpu, device, sup.memoryBudget);
//   TrxGpu::Buffer buf;
//   alloc.CreateBuffer(1 << 20, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, TrxGpu::MemoryUsage::BarUpload, buf);

#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace TrxGpu {

// ---------------------------------------------------------------------------------------------
// 1. Instancja / urzadzenie: rozszerzenia i funkcje
// ---------------------------------------------------------------------------------------------

// Dodaj do listy rozszerzen instancji i flag. Na macOS (MoltenVK) bez tego instancja nie
// zobaczy zadnego GPU. Na pozostalych systemach nic nie zmienia.
inline void AppendInstancePortability(std::vector<const char*>& extensions, VkInstanceCreateFlags& flags) {
#ifdef VK_INSTANCE_CREATE_ENABLE_PORTABILITY_BIT_KHR
	uint32_t n = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
	std::vector<VkExtensionProperties> props(n);
	vkEnumerateInstanceExtensionProperties(nullptr, &n, props.data());

	const bool available = std::any_of(props.begin(), props.end(), [](const VkExtensionProperties& p) {
		return std::strcmp(p.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
	});

	if (available) {
		extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		flags |= VK_INSTANCE_CREATE_ENABLE_PORTABILITY_BIT_KHR;
	}
#else
	(void)extensions;
	(void)flags;
#endif
}

inline bool HasDeviceExtension(VkPhysicalDevice gpu, const char* name) {
	uint32_t n = 0;
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n, nullptr);
	std::vector<VkExtensionProperties> props(n);
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &n, props.data());

	return std::any_of(props.begin(), props.end(),
	                   [&](const VkExtensionProperties& p) { return std::strcmp(p.extensionName, name) == 0; });
}

struct DeviceSupport {
	std::vector<const char*> extensions;       // dodaj do VkDeviceCreateInfo::ppEnabledExtensionNames
	bool                     memoryBudget      = false;
	bool                     timelineSemaphore = false; // wlacz VkPhysicalDeviceVulkan12Features::timelineSemaphore
};

inline DeviceSupport QueryDeviceSupport(VkPhysicalDevice gpu) {
	DeviceSupport s;

	if (HasDeviceExtension(gpu, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
		s.memoryBudget = true;
		s.extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
	}

	// Spec wymaga wlaczenia tego rozszerzenia, jesli urzadzenie je udostepnia (MoltenVK).
	if (HasDeviceExtension(gpu, "VK_KHR_portability_subset")) {
		s.extensions.push_back("VK_KHR_portability_subset");
	}

	VkPhysicalDeviceProperties props {};
	vkGetPhysicalDeviceProperties(gpu, &props);

	if (props.apiVersion >= VK_API_VERSION_1_2) {
		VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
		VkPhysicalDeviceFeatures2        f2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
		vkGetPhysicalDeviceFeatures2(gpu, &f2);
		s.timelineSemaphore = f12.timelineSemaphore == VK_TRUE;
	}

	return s;
}

// ---------------------------------------------------------------------------------------------
// 2. Rodziny kolejek
// ---------------------------------------------------------------------------------------------

struct QueueFamilies {
	uint32_t graphics = UINT32_MAX;
	uint32_t compute  = UINT32_MAX; // dedykowana, jesli jest; inaczej = graphics
	uint32_t transfer = UINT32_MAX; // dedykowana (DMA), jesli jest; inaczej = graphics

	bool HasDedicatedCompute() const { return compute != graphics; }
	bool HasDedicatedTransfer() const { return transfer != graphics; }
};

// Uwaga: nie sprawdza obslugi prezentacji (to wymaga VkSurfaceKHR) - zrob to osobno dla graphics.
inline std::optional<QueueFamilies> PickQueueFamilies(VkPhysicalDevice gpu) {
	uint32_t n = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &n, nullptr);
	std::vector<VkQueueFamilyProperties> props(n);
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &n, props.data());

	QueueFamilies q;

	for (uint32_t i = 0; i < n; i++) {
		if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			q.graphics = i;
			break;
		}
	}

	if (q.graphics == UINT32_MAX) {
		return std::nullopt;
	}

	q.compute  = q.graphics;
	q.transfer = q.graphics;

	for (uint32_t i = 0; i < n; i++) {
		const VkQueueFlags f = props[i].queueFlags;

		if ((f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT) && q.compute == q.graphics) {
			q.compute = i;
		}

		if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) &&
		    q.transfer == q.graphics) {
			q.transfer = i;
		}
	}

	return q;
}

// Pula komend jednego watku. Pule NIE sa thread-safe - kazdy watek ma miec wlasna.
inline VkResult CreateCommandPool(VkDevice device, uint32_t family, bool transient, VkCommandPool& out) {
	VkCommandPoolCreateInfo ci {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
	           (transient ? VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : 0);
	ci.queueFamilyIndex = family;
	return vkCreateCommandPool(device, &ci, nullptr, &out);
}

// ---------------------------------------------------------------------------------------------
// 3. Semafory timeline i Submit
// ---------------------------------------------------------------------------------------------

inline VkResult CreateTimelineSemaphore(VkDevice device, uint64_t initial, VkSemaphore& out) {
	VkSemaphoreTypeCreateInfo type {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
	type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	type.initialValue  = initial;

	VkSemaphoreCreateInfo ci {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
	return vkCreateSemaphore(device, &ci, nullptr, &out);
}

// VK_SUCCESS = osiagnieto wartosc, VK_TIMEOUT = nie doczekano sie.
inline VkResult WaitTimeline(VkDevice device, VkSemaphore sem, uint64_t value, uint64_t timeout_ns) {
	VkSemaphoreWaitInfo wi {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
	wi.semaphoreCount = 1;
	wi.pSemaphores    = &sem;
	wi.pValues        = &value;
	return vkWaitSemaphores(device, &wi, timeout_ns);
}

struct TimelineWait {
	VkSemaphore          sem;
	uint64_t             value;
	VkPipelineStageFlags stage;
};

// Jeden vkQueueSubmit dla calej paczki komend (mniej wywolan = mniej narzutu CPU).
// Bez alokacji na stercie; do 8 oczekiwan.
inline VkResult SubmitTimeline(VkQueue queue, std::span<const VkCommandBuffer> cmds,
                               std::span<const TimelineWait> waits, VkSemaphore signal_sem,
                               uint64_t signal_value, VkFence fence = VK_NULL_HANDLE) {
	constexpr size_t kMaxWaits = 8;
	if (waits.size() > kMaxWaits) {
		return VK_ERROR_TOO_MANY_OBJECTS;
	}

	std::array<VkSemaphore, kMaxWaits>          wait_sems {};
	std::array<uint64_t, kMaxWaits>             wait_values {};
	std::array<VkPipelineStageFlags, kMaxWaits> wait_stages {};

	for (size_t i = 0; i < waits.size(); i++) {
		wait_sems[i]   = waits[i].sem;
		wait_values[i] = waits[i].value;
		wait_stages[i] = waits[i].stage;
	}

	VkTimelineSemaphoreSubmitInfo ts {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
	ts.waitSemaphoreValueCount   = static_cast<uint32_t>(waits.size());
	ts.pWaitSemaphoreValues      = wait_values.data();
	ts.signalSemaphoreValueCount = signal_sem != VK_NULL_HANDLE ? 1 : 0;
	ts.pSignalSemaphoreValues    = &signal_value;

	VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts};
	si.waitSemaphoreCount   = static_cast<uint32_t>(waits.size());
	si.pWaitSemaphores      = wait_sems.data();
	si.pWaitDstStageMask    = wait_stages.data();
	si.commandBufferCount   = static_cast<uint32_t>(cmds.size());
	si.pCommandBuffers      = cmds.data();
	si.signalSemaphoreCount = signal_sem != VK_NULL_HANDLE ? 1 : 0;
	si.pSignalSemaphores    = &signal_sem;

	return vkQueueSubmit(queue, 1, &si, fence);
}

// ---------------------------------------------------------------------------------------------
// 4. Pamiec: wykrywanie ReBAR / UMA i alokator buforow
// ---------------------------------------------------------------------------------------------

struct MemoryInfo {
	VkDeviceSize deviceLocalBytes = 0; // najwiekszy heap DEVICE_LOCAL
	VkDeviceSize barBytes         = 0; // najwiekszy heap z typem DEVICE_LOCAL|HOST_VISIBLE
	bool         unified          = false; // wszystkie heapy DEVICE_LOCAL (iGPU, Apple)
	bool         resizableBar     = false; // BAR wyraznie wiekszy niz klasyczne okno 256 MiB
};

inline MemoryInfo QueryMemoryInfo(const VkPhysicalDeviceMemoryProperties& mp) {
	MemoryInfo info;
	info.unified = mp.memoryHeapCount > 0;

	for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
		if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			info.deviceLocalBytes = std::max(info.deviceLocalBytes, mp.memoryHeaps[i].size);
		} else {
			info.unified = false;
		}
	}

	constexpr VkMemoryPropertyFlags kBar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
		if ((mp.memoryTypes[i].propertyFlags & kBar) == kBar) {
			info.barBytes = std::max(info.barBytes, mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size);
		}
	}

	info.resizableBar = !info.unified && info.barBytes > (256ull << 20);
	return info;
}

enum class MemoryUsage {
	DeviceOnly, // tylko GPU (tekstury, bufory wierzcholkow) - najszybsze dla GPU
	Staging,    // CPU -> GPU przez kopie; NIE zajmuje okna BAR
	BarUpload,  // zapis CPU bezposrednio do VRAM (ReBAR/UMA); fallback = Staging
	Readback,   // GPU -> CPU, pamiec cached
};

struct Buffer {
	VkBuffer              buffer     = VK_NULL_HANDLE;
	VkDeviceMemory        memory     = VK_NULL_HANDLE;
	VkDeviceSize          size       = 0;
	VkDeviceSize          allocSize  = 0;
	void*                 mapped     = nullptr; // nullptr dla DeviceOnly
	VkMemoryPropertyFlags flags      = 0;

	bool Coherent() const { return (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0; }
	bool DirectVram() const {
		return (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	}
};

class Allocator {
public:
	Allocator(VkPhysicalDevice gpu, VkDevice device, bool budget_extension_enabled)
	    : m_gpu(gpu), m_device(device), m_budget(budget_extension_enabled) {
		vkGetPhysicalDeviceMemoryProperties(gpu, &m_mem);

		VkPhysicalDeviceProperties props {};
		vkGetPhysicalDeviceProperties(gpu, &props);
		m_atom = std::max<VkDeviceSize>(props.limits.nonCoherentAtomSize, 1);
	}

	MemoryInfo Info() const { return QueryMemoryInfo(m_mem); }

	VkResult CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, MemoryUsage mu, Buffer& out) {
		out = {};
		if (size == 0) {
			return VK_ERROR_INITIALIZATION_FAILED;
		}

		VkBufferCreateInfo ci {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		ci.size        = size;
		ci.usage       = usage;
		ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkBuffer buf = VK_NULL_HANDLE;
		if (VkResult r = vkCreateBuffer(m_device, &ci, nullptr, &buf); r != VK_SUCCESS) {
			return r;
		}

		VkMemoryRequirements req {};
		vkGetBufferMemoryRequirements(m_device, buf, &req);

		VkResult last = VK_ERROR_OUT_OF_DEVICE_MEMORY;

		for (const Preference& pref: Preferences(mu)) {
			const auto type = FindType(req.memoryTypeBits, pref.want, pref.forbid);
			if (!type) {
				continue;
			}

			const uint32_t heap = m_mem.memoryTypes[*type].heapIndex;
			if (!HeapHasRoom(heap, req.size)) {
				last = VK_ERROR_OUT_OF_DEVICE_MEMORY;
				continue;
			}

			VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
			ai.allocationSize  = req.size;
			ai.memoryTypeIndex = *type;

			VkDeviceMemory mem = VK_NULL_HANDLE;
			if (last = vkAllocateMemory(m_device, &ai, nullptr, &mem); last != VK_SUCCESS) {
				continue;
			}

			if (last = vkBindBufferMemory(m_device, buf, mem, 0); last != VK_SUCCESS) {
				vkFreeMemory(m_device, mem, nullptr);
				continue;
			}

			const VkMemoryPropertyFlags flags = m_mem.memoryTypes[*type].propertyFlags;
			void*                       ptr   = nullptr;

			if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
				if (last = vkMapMemory(m_device, mem, 0, VK_WHOLE_SIZE, 0, &ptr); last != VK_SUCCESS) {
					vkFreeMemory(m_device, mem, nullptr);
					continue;
				}
			}

			out = {buf, mem, size, req.size, ptr, flags};
			return VK_SUCCESS;
		}

		vkDestroyBuffer(m_device, buf, nullptr);
		return last;
	}

	void DestroyBuffer(Buffer& b) {
		if (b.mapped) {
			vkUnmapMemory(m_device, b.memory);
		}
		if (b.buffer != VK_NULL_HANDLE) {
			vkDestroyBuffer(m_device, b.buffer, nullptr);
		}
		if (b.memory != VK_NULL_HANDLE) {
			vkFreeMemory(m_device, b.memory, nullptr);
		}
		b = {};
	}

	// Po zapisie CPU do pamieci NIE-coherent (na coherent nic nie robi).
	VkResult Flush(const Buffer& b, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const {
		if (!b.mapped || b.Coherent()) {
			return VK_SUCCESS;
		}
		const VkMappedMemoryRange r = Range(b, offset, size);
		return vkFlushMappedMemoryRanges(m_device, 1, &r);
	}

	// Przed odczytem CPU danych zapisanych przez GPU (na coherent nic nie robi).
	VkResult Invalidate(const Buffer& b, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE) const {
		if (!b.mapped || b.Coherent()) {
			return VK_SUCCESS;
		}
		const VkMappedMemoryRange r = Range(b, offset, size);
		return vkInvalidateMappedMemoryRanges(m_device, 1, &r);
	}

private:
	struct Preference {
		VkMemoryPropertyFlags want;
		VkMemoryPropertyFlags forbid;
	};

	static constexpr VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	static constexpr VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
	static constexpr VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	static constexpr VkMemoryPropertyFlags CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

	static std::vector<Preference> Preferences(MemoryUsage mu) {
		switch (mu) {
			case MemoryUsage::DeviceOnly: return {{DL, 0}};
			case MemoryUsage::Staging: return {{HV | HC, DL}, {HV | HC, 0}};
			case MemoryUsage::BarUpload: return {{DL | HV | HC, 0}, {HV | HC, DL}, {HV | HC, 0}};
			case MemoryUsage::Readback: return {{HV | CA, DL}, {HV | HC, DL}, {HV | HC, 0}};
		}
		return {};
	}

	std::optional<uint32_t> FindType(uint32_t type_bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags forbid) const {
		for (uint32_t i = 0; i < m_mem.memoryTypeCount; i++) {
			const VkMemoryPropertyFlags f = m_mem.memoryTypes[i].propertyFlags;
			if ((type_bits & (1u << i)) && (f & want) == want && (f & forbid) == 0) {
				return i;
			}
		}
		return std::nullopt;
	}

	// Odmawia alokacji, ktora przekroczylaby budzet heapu (zamiast ryzykowac paging / TDR).
	bool HeapHasRoom(uint32_t heap, VkDeviceSize size) const {
		if (!m_budget) {
			return true;
		}

		VkPhysicalDeviceMemoryBudgetPropertiesEXT budget {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
		VkPhysicalDeviceMemoryProperties2         props {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &budget};
		vkGetPhysicalDeviceMemoryProperties2(m_gpu, &props);

		const VkDeviceSize cap  = budget.heapBudget[heap];
		const VkDeviceSize used = budget.heapUsage[heap];
		return used <= cap && size <= cap - used;
	}

	VkMappedMemoryRange Range(const Buffer& b, VkDeviceSize offset, VkDeviceSize size) const {
		VkMappedMemoryRange r {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
		r.memory = b.memory;
		r.offset = offset / m_atom * m_atom;

		if (size == VK_WHOLE_SIZE) {
			r.size = VK_WHOLE_SIZE;
		} else {
			const VkDeviceSize end = (offset + size + m_atom - 1) / m_atom * m_atom;
			r.size                 = std::min(end, b.allocSize) - r.offset;
		}
		return r;
	}

	VkPhysicalDevice                 m_gpu;
	VkDevice                         m_device;
	bool                             m_budget;
	VkDeviceSize                     m_atom = 1;
	VkPhysicalDeviceMemoryProperties m_mem {};
};

// ---------------------------------------------------------------------------------------------
// 5. Pipeline cache na dysku (mniej zacinania przy kompilacji pipeline'ow)
// ---------------------------------------------------------------------------------------------

namespace detail {
struct PipelineCacheHeader {
	uint32_t headerSize;
	uint32_t headerVersion;
	uint32_t vendorID;
	uint32_t deviceID;
	uint8_t  uuid[VK_UUID_SIZE];
};
static_assert(sizeof(PipelineCacheHeader) == 16 + VK_UUID_SIZE);
} // namespace detail

// Wczytuje cache tylko jesli naglowek pasuje do tego GPU i sterownika; inaczej startuje od zera.
inline VkResult LoadPipelineCache(VkDevice device, const VkPhysicalDeviceProperties& props,
                                  const std::filesystem::path& file, VkPipelineCache& out) {
	std::vector<char> data;

	if (std::ifstream in(file, std::ios::binary | std::ios::ate); in) {
		const std::streamoff size = in.tellg();
		if (size >= static_cast<std::streamoff>(sizeof(detail::PipelineCacheHeader))) {
			data.resize(static_cast<size_t>(size));
			in.seekg(0);
			in.read(data.data(), size);
			if (!in) {
				data.clear();
			}
		}
	}

	if (!data.empty()) {
		detail::PipelineCacheHeader h {};
		std::memcpy(&h, data.data(), sizeof(h));

		const bool valid = h.headerSize >= sizeof(h) && h.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
		                   h.vendorID == props.vendorID && h.deviceID == props.deviceID &&
		                   std::memcmp(h.uuid, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
		if (!valid) {
			data.clear();
		}
	}

	VkPipelineCacheCreateInfo ci {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
	ci.initialDataSize = data.size();
	ci.pInitialData    = data.empty() ? nullptr : data.data();
	return vkCreatePipelineCache(device, &ci, nullptr, &out);
}

// Zapis przez plik tymczasowy + rename: awaria w trakcie nie zostawi uszkodzonego cache.
inline bool SavePipelineCache(VkDevice device, VkPipelineCache cache, const std::filesystem::path& file) {
	size_t size = 0;
	if (vkGetPipelineCacheData(device, cache, &size, nullptr) != VK_SUCCESS || size == 0) {
		return false;
	}

	std::vector<char> data(size);
	if (vkGetPipelineCacheData(device, cache, &size, data.data()) < 0) {
		return false;
	}
	data.resize(size);

	std::filesystem::path tmp = file;
	tmp += ".tmp";

	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if (!out) {
			return false;
		}
		out.write(data.data(), static_cast<std::streamsize>(data.size()));
		if (!out) {
			return false;
		}
	}

	std::error_code ec;
	std::filesystem::rename(tmp, file, ec);
	if (ec) {
		std::filesystem::remove(tmp, ec);
		return false;
	}
	return true;
}

} // namespace TrxGpu

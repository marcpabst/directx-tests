use thread_priority::{set_current_thread_priority, ThreadPriority};
use windows::{
    core::*,
    Wdk::Graphics::Direct3D::*,
    Win32::{
        Foundation::*,
        Graphics::{
            Direct3D::{Fxc::*, *},
            Direct3D12::*,
            Dxgi::{Common::*, *},
            Gdi::{
                CreateDCW, DeleteDC, EnumDisplayDevicesW, DISPLAY_DEVICEW,
                DISPLAY_DEVICE_PRIMARY_DEVICE,
            },
        },
        System::{LibraryLoader::*, Performance::*, Threading::*},
        UI::WindowsAndMessaging::*,
    },
};

use std::mem::transmute;

mod refresh_rates;

// set up a static mutex containing a instant
lazy_static::lazy_static! {
    static ref LAST_TIME: std::sync::Mutex<std::time::Instant> = std::sync::Mutex::new(std::time::Instant::now());
    static ref LAST_INTERUPT_TIMESTAMP: std::sync::Mutex<i64> = std::sync::Mutex::new(0);
    static ref LAST_VLBANK_TIMESTAMP: std::sync::Mutex<i64> = std::sync::Mutex::new(0);
    static ref LAST_FLIP: std::sync::Mutex<u32> = std::sync::Mutex::new(0);
}

// macro to time funsction calls
// example usage:
// time! {
//     let x = 1 + 1;
// }
macro_rules! time {
    ($block:block) => {{
        let start = std::time::Instant::now();
        let result = { $block };
        let duration = start.elapsed();
        log::info!("Time elapsed: {:?}", duration);
        result
    }};
}

fn variance(data: &Vec<f64>) -> f64 {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance
}

fn std_dev(data: &Vec<f64>) -> f64 {
    variance(data).sqrt()
}

fn mean(data: &Vec<f64>) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn mininum(data: &Vec<f64>) -> f64 {
    *data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

fn maximum(data: &Vec<f64>) -> f64 {
    *data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

fn report_stats(data: &Vec<f64>, name: &str) {
    let mean = mean(data);
    let std_dev = std_dev(data);
    let variance = variance(data);
    let min = mininum(data);
    let max = maximum(data);
    let n = data.len();

    log::info!(
        "{}: mean: {:.5} std_dev: {:.5} variance: {:.5}, range: {:.5}..{:.5} (n={})",
        name,
        mean,
        std_dev,
        variance,
        min,
        max,
        n
    );
}

fn get_vblank_handle() -> Option<D3DKMT_WAITFORVERTICALBLANKEVENT> {
    let mut dd = DISPLAY_DEVICEW::default();
    dd.cb = std::mem::size_of::<DISPLAY_DEVICEW>() as u32;

    let mut device_num = 0;
    while unsafe { EnumDisplayDevicesW(None, device_num, &mut dd, 0).as_bool() } {
        // DumpDevice(dd, 0); // You can implement this function if needed

        if dd.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE != 0 {
            break;
        }

        device_num += 1;
    }

    // convert device name to a HSRTING
    let device_name: [u16; 32] = dd.DeviceName;
    let device_name = device_name
        .iter()
        .copied()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let device_name: String = String::from_utf16(&device_name).unwrap();
    let device_name: &HSTRING = &HSTRING::from(device_name);

    let hdc = unsafe { CreateDCW(None, device_name, None, None) };
    if hdc.is_invalid() {
        return None;
    }

    let mut open_adapter_data = D3DKMT_OPENADAPTERFROMHDC::default();
    open_adapter_data.hDc = hdc;

    if unsafe { D3DKMTOpenAdapterFromHdc(&mut open_adapter_data) } == STATUS_SUCCESS {
        unsafe { DeleteDC(hdc) };
    } else {
        unsafe { DeleteDC(hdc) };
        return None;
    }

    let mut we = D3DKMT_WAITFORVERTICALBLANKEVENT::default();
    we.hAdapter = open_adapter_data.hAdapter;
    we.hDevice = 0; // Optional: You can use OpenDeviceHandle to get the device handle
    we.VidPnSourceId = open_adapter_data.VidPnSourceId;

    Some(we)
}

trait DXSample {
    fn new(command_line: &SampleCommandLine) -> Result<Self>
    where
        Self: Sized;

    fn bind_to_window(&mut self, hwnd: &HWND) -> Result<()>;

    fn update(&mut self) {}
    fn render(&mut self) {}
    fn on_key_up(&mut self, _key: u8) {}
    fn on_key_down(&mut self, _key: u8) {}

    fn title(&self) -> String {
        "DXSample".into()
    }

    fn window_size(&self) -> (i32, i32) {
        (640, 480)
    }
}

#[derive(Clone)]
struct SampleCommandLine {
    use_warp_device: bool,
}

fn build_command_line() -> SampleCommandLine {
    let mut use_warp_device = false;

    for arg in std::env::args() {
        if arg.eq_ignore_ascii_case("-warp") || arg.eq_ignore_ascii_case("/warp") {
            use_warp_device = true;
        }
    }

    SampleCommandLine { use_warp_device }
}

fn run_sample<S>() -> Result<()>
where
    S: DXSample,
{
    let instance = unsafe { GetModuleHandleA(None)? };

    let wc = WNDCLASSEXA {
        cbSize: std::mem::size_of::<WNDCLASSEXA>() as u32,
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wndproc::<S>),
        hInstance: instance.into(),
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW)? },
        lpszClassName: s!("RustWindowClass"),
        ..Default::default()
    };

    let command_line = build_command_line();
    let mut sample = S::new(&command_line)?;

    let size = sample.window_size();

    let atom = unsafe { RegisterClassExA(&wc) };
    debug_assert_ne!(atom, 0);

    let mut window_rect = RECT {
        left: 0,
        top: 0,
        right: size.0,
        bottom: size.1,
    };
    unsafe { AdjustWindowRect(&mut window_rect, WS_POPUP, false)? };

    let mut title = sample.title();

    if command_line.use_warp_device {
        title.push_str(" (WARP)");
    }

    title.push('\0');

    let hwnd = unsafe {
        CreateWindowExA(
            WINDOW_EX_STYLE::default(),
            s!("RustWindowClass"),
            PCSTR(title.as_ptr()),
            WS_POPUP,
            0,
            0,
            GetSystemMetrics(SM_CXSCREEN),
            GetSystemMetrics(SM_CYSCREEN),
            None, // no parent windowf
            None, // no menus
            instance,
            Some(&mut sample as *mut _ as _),
        )
    };

    sample.bind_to_window(&hwnd)?;

    log::info!("hello from run_sample 3");

    unsafe { _ = ShowWindow(hwnd, SW_SHOW) };

    log::info!("hello from run_sample 4");

    loop {
        let mut message = MSG::default();

        if unsafe { PeekMessageA(&mut message, None, 0, 0, PM_REMOVE) }.into() {
            unsafe {
                _ = TranslateMessage(&message);
                DispatchMessageA(&message);
            }

            if message.message == WM_QUIT {
                break;
            }
        }
    }

    Ok(())
}

fn sample_wndproc<S: DXSample>(sample: &mut S, message: u32, wparam: WPARAM) -> bool {
    match message {
        WM_KEYDOWN => {
            sample.on_key_down(wparam.0 as u8);
            true
        }
        WM_KEYUP => {
            sample.on_key_up(wparam.0 as u8);
            true
        }
        WM_PAINT => {
            sample.update();
            sample.render();
            true
        }
        _ => false,
    }
}

extern "system" fn wndproc<S: DXSample>(
    window: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match message {
        WM_CREATE => {
            unsafe {
                let create_struct: &CREATESTRUCTA = transmute(lparam);
                SetWindowLongPtrA(window, GWLP_USERDATA, create_struct.lpCreateParams as _);
            }
            LRESULT::default()
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            LRESULT::default()
        }
        _ => {
            let user_data = unsafe { GetWindowLongPtrA(window, GWLP_USERDATA) };
            let sample = std::ptr::NonNull::<S>::new(user_data as _);
            let handled = sample.map_or(false, |mut s| {
                sample_wndproc(unsafe { s.as_mut() }, message, wparam)
            });

            if handled {
                LRESULT::default()
            } else {
                unsafe { DefWindowProcA(window, message, wparam, lparam) }
            }
        }
    }
}

fn get_hardware_adapter(factory: &IDXGIFactory4) -> Result<IDXGIAdapter1> {
    for i in 0.. {
        let adapter = unsafe { factory.EnumAdapters1(i)? };

        let mut desc = Default::default();
        unsafe { adapter.GetDesc1(&mut desc)? };

        if (DXGI_ADAPTER_FLAG(desc.Flags as i32) & DXGI_ADAPTER_FLAG_SOFTWARE)
            != DXGI_ADAPTER_FLAG_NONE
        {
            // Don't select the Basic Render Driver adapter. If you want a
            // software adapter, pass in "/warp" on the command line.
            continue;
        }

        // Check to see whether the adapter supports Direct3D 12, but don't
        // create the actual device yet.
        if unsafe {
            D3D12CreateDevice(
                &adapter,
                D3D_FEATURE_LEVEL_11_0,
                std::ptr::null_mut::<Option<ID3D12Device>>(),
            )
        }
        .is_ok()
        {
            return Ok(adapter);
        }
    }

    unreachable!()
}

mod d3d12_hello_triangle {

    use std::f32::consts::E;

    use super::*;

    const FRAME_COUNT: u32 = 2;

    pub struct Sample {
        dxgi_factory: IDXGIFactory4,
        device: ID3D12Device,
        adapter: IDXGIAdapter1,
        resources: Option<Resources>,
    }

    struct Resources {
        command_queue: ID3D12CommandQueue,
        swap_chain: IDXGISwapChain3,
        adapter: IDXGIAdapter1,
        wait_for_vblank_event: D3DKMT_WAITFORVERTICALBLANKEVENT,
        frame_index: u32,
        render_targets: [ID3D12Resource; FRAME_COUNT as usize],
        rtv_heap: ID3D12DescriptorHeap,
        rtv_descriptor_size: usize,
        viewport: D3D12_VIEWPORT,
        scissor_rect: RECT,
        command_allocator: ID3D12CommandAllocator,
        root_signature: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        command_list: ID3D12GraphicsCommandList,

        // we need to keep this around to keep the reference alive, even though
        // nothing reads from it
        #[allow(dead_code)]
        vertex_buffer: ID3D12Resource,

        vbv: D3D12_VERTEX_BUFFER_VIEW,
        fence: ID3D12Fence,
        fence_value: u64,
        fence_event: HANDLE,

        frame_rate_calc: refresh_rates::RefreshRateCalculator,
        frame_times_wait_for_vblank: Vec<f64>,
        frame_times_interupt: Vec<f64>,
    }

    impl DXSample for Sample {
        fn new(command_line: &SampleCommandLine) -> Result<Self> {
            let (dxgi_factory, device, adapter) = create_device(command_line)?;

            Ok(Sample {
                dxgi_factory,
                device,
                adapter,
                resources: None,
            })
        }

        fn bind_to_window(&mut self, hwnd: &HWND) -> Result<()> {
            let command_queue: ID3D12CommandQueue = unsafe {
                self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?
            };

            let (width, height) = self.window_size();

            let adapter = self.adapter.clone();

            let swap_chain_desc = DXGI_SWAP_CHAIN_DESC1 {
                BufferCount: FRAME_COUNT,
                Width: width as u32,
                Height: height as u32,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };

            // exclusive fullscreen
            let pfullscreendesc = DXGI_SWAP_CHAIN_FULLSCREEN_DESC {
                RefreshRate: DXGI_RATIONAL {
                    Numerator: 120,
                    Denominator: 1,
                },
                Windowed: false.into(),
                ..Default::default()
            };
            let swap_chain: IDXGISwapChain3 = unsafe {
                self.dxgi_factory.CreateSwapChainForHwnd(
                    &command_queue,
                    *hwnd,
                    &swap_chain_desc,
                    Some(&pfullscreendesc),
                    None,
                )?
            }
            .cast()?;

            // // Set fullscreen state
            // unsafe {
            //     swap_chain.SetFullscreenState(true, None)?;
            // }

            // This sample does not support fullscreen transitions
            unsafe {
                self.dxgi_factory
                    .MakeWindowAssociation(*hwnd, DXGI_MWA_NO_ALT_ENTER)?;
            }

            let frame_index = unsafe { swap_chain.GetCurrentBackBufferIndex() };

            let rtv_heap: ID3D12DescriptorHeap = unsafe {
                self.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        NumDescriptors: FRAME_COUNT,
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        ..Default::default()
                    })
            }?;

            let rtv_descriptor_size = unsafe {
                self.device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
            } as usize;
            let rtv_handle = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };

            let render_targets: [ID3D12Resource; FRAME_COUNT as usize] =
                array_init::try_array_init(|i: usize| -> Result<ID3D12Resource> {
                    let render_target: ID3D12Resource = unsafe { swap_chain.GetBuffer(i as u32) }?;
                    unsafe {
                        self.device.CreateRenderTargetView(
                            &render_target,
                            None,
                            D3D12_CPU_DESCRIPTOR_HANDLE {
                                ptr: rtv_handle.ptr + i * rtv_descriptor_size,
                            },
                        )
                    };
                    Ok(render_target)
                })?;

            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: D3D12_MIN_DEPTH,
                MaxDepth: D3D12_MAX_DEPTH,
            };

            let scissor_rect = RECT {
                left: 0,
                top: 0,
                right: width,
                bottom: height,
            };

            let command_allocator = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            }?;

            let root_signature = create_root_signature(&self.device)?;
            let pso = create_pipeline_state(&self.device, &root_signature)?;

            let command_list: ID3D12GraphicsCommandList = unsafe {
                self.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &command_allocator,
                    &pso,
                )
            }?;
            unsafe {
                command_list.Close()?;
            };

            let aspect_ratio = width as f32 / height as f32;

            let (vertex_buffer, vbv) = create_vertex_buffer(&self.device, aspect_ratio)?;

            let fence = unsafe { self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }?;

            let fence_value = 1;

            let fence_event = unsafe { CreateEventA(None, false, false, None)? };

            log::info!("hello from bind_to_window");

            self.resources = Some(Resources {
                command_queue,
                swap_chain,
                adapter: adapter,
                wait_for_vblank_event: get_vblank_handle().unwrap(),
                frame_index,
                render_targets,
                rtv_heap,
                rtv_descriptor_size,
                viewport,
                scissor_rect,
                command_allocator,
                root_signature,
                pso,
                command_list,
                vertex_buffer,
                vbv,
                fence,
                fence_value,
                fence_event,
                frame_rate_calc: refresh_rates::RefreshRateCalculator::new(),
                frame_times_interupt: vec![],
                frame_times_wait_for_vblank: vec![],
            });

            log::info!("hello from bind_to_window 2");

            Ok(())
        }

        fn title(&self) -> String {
            "D3D12 Hello Triangle".into()
        }

        fn window_size(&self) -> (i32, i32) {
            (1280, 720)
        }

        fn render(&mut self) {
            if let Some(resources) = &mut self.resources {
                populate_command_list(resources).unwrap();

                // Execute the command list.
                let command_list = Some(resources.command_list.cast().unwrap());
                unsafe { resources.command_queue.ExecuteCommandLists(&[command_list]) };

                // simulate some work (busy wait for 1ms)
                let start = std::time::Instant::now();
                while start.elapsed().as_millis() < 1 {}

                // Present the frame.x
                unsafe { resources.swap_chain.Present(1, 0) }.ok().unwrap();

                wait_for_previous_frame(resources);
            }
        }
    }

    fn populate_command_list(resources: &Resources) -> Result<()> {
        // Command list allocators can only be reset when the associated
        // command lists have finished execution on the GPU; apps should use
        // fences to determine GPU execution progress.
        unsafe {
            resources.command_allocator.Reset()?;
        }

        let command_list = &resources.command_list;

        // However, when ExecuteCommandList() is called on a particular
        // command list, that command list can then be reset at any time and
        // must be before re-recording.
        unsafe {
            command_list.Reset(&resources.command_allocator, &resources.pso)?;
        }

        // Set necessary state.
        unsafe {
            command_list.SetGraphicsRootSignature(&resources.root_signature);
            command_list.RSSetViewports(&[resources.viewport]);
            command_list.RSSetScissorRects(&[resources.scissor_rect]);
        }

        // Indicate that the back buffer will be used as a render target.
        let barrier = transition_barrier(
            &resources.render_targets[resources.frame_index as usize],
            D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe { command_list.ResourceBarrier(&[barrier]) };

        let rtv_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: unsafe { resources.rtv_heap.GetCPUDescriptorHandleForHeapStart() }.ptr
                + resources.frame_index as usize * resources.rtv_descriptor_size,
        };

        unsafe { command_list.OMSetRenderTargets(1, Some(&rtv_handle), false, None) };

        // Record commands.
        unsafe {
            command_list.ClearRenderTargetView(
                rtv_handle,
                &[0.0_f32, 0.2_f32, 0.4_f32, 1.0_f32],
                None,
            );
            command_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            command_list.IASetVertexBuffers(0, Some(&[resources.vbv]));
            command_list.DrawInstanced(3, 1, 0, 0);

            // Indicate that the back buffer will now be used to present.
            command_list.ResourceBarrier(&[transition_barrier(
                &resources.render_targets[resources.frame_index as usize],
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            )]);
        }

        unsafe { command_list.Close() }
    }

    fn transition_barrier(
        resource: &ID3D12Resource,
        state_before: D3D12_RESOURCE_STATES,
        state_after: D3D12_RESOURCE_STATES,
    ) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(resource) },
                    StateBefore: state_before,
                    StateAfter: state_after,
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                }),
            },
        }
    }

    fn create_device(
        command_line: &SampleCommandLine,
    ) -> Result<(IDXGIFactory4, ID3D12Device, IDXGIAdapter1)> {
        if cfg!(debug_assertions) {
            unsafe {
                let mut debug: Option<ID3D12Debug> = None;
                if let Some(debug) = D3D12GetDebugInterface(&mut debug).ok().and(debug) {
                    debug.EnableDebugLayer();
                }
            }
        }

        let dxgi_factory_flags = if cfg!(debug_assertions) {
            DXGI_CREATE_FACTORY_DEBUG
        } else {
            0
        };

        let dxgi_factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(dxgi_factory_flags) }?;

        let adapter = if command_line.use_warp_device {
            unsafe { dxgi_factory.EnumWarpAdapter() }
        } else {
            get_hardware_adapter(&dxgi_factory)
        }?;

        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }?;
        Ok((dxgi_factory, device.unwrap(), adapter))
    }

    fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            ..Default::default()
        };

        let mut signature = None;

        let signature = unsafe {
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut signature, None)
        }
        .map(|()| signature.unwrap())?;

        unsafe {
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    signature.GetBufferPointer() as _,
                    signature.GetBufferSize(),
                ),
            )
        }
    }

    fn create_pipeline_state(
        device: &ID3D12Device,
        root_signature: &ID3D12RootSignature,
    ) -> Result<ID3D12PipelineState> {
        let compile_flags = if cfg!(debug_assertions) {
            D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION
        } else {
            0
        };

        let exe_path = std::env::current_exe().ok().unwrap();
        let asset_path = exe_path.parent().unwrap();
        let shaders_hlsl_path = asset_path.join("shaders.hlsl");
        let shaders_hlsl = shaders_hlsl_path.to_str().unwrap();
        let shaders_hlsl: HSTRING = shaders_hlsl.into();

        let mut vertex_shader = None;
        let vertex_shader = unsafe {
            D3DCompileFromFile(
                &shaders_hlsl,
                None,
                None,
                s!("VSMain"),
                s!("vs_5_0"),
                compile_flags,
                0,
                &mut vertex_shader,
                None,
            )
        }
        .map(|()| vertex_shader.unwrap())?;

        let mut pixel_shader = None;
        let pixel_shader = unsafe {
            D3DCompileFromFile(
                &shaders_hlsl,
                None,
                None,
                s!("PSMain"),
                s!("ps_5_0"),
                compile_flags,
                0,
                &mut pixel_shader,
                None,
            )
        }
        .map(|()| pixel_shader.unwrap())?;

        let mut input_element_descs: [D3D12_INPUT_ELEMENT_DESC; 2] = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("COLOR"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];

        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: input_element_descs.as_mut_ptr(),
                NumElements: input_element_descs.len() as u32,
            },
            pRootSignature: unsafe { std::mem::transmute_copy(root_signature) },
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: unsafe { vertex_shader.GetBufferPointer() },
                BytecodeLength: unsafe { vertex_shader.GetBufferSize() },
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: unsafe { pixel_shader.GetBufferPointer() },
                BytecodeLength: unsafe { pixel_shader.GetBufferSize() },
            },
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                ..Default::default()
            },
            BlendState: D3D12_BLEND_DESC {
                AlphaToCoverageEnable: false.into(),
                IndependentBlendEnable: false.into(),
                RenderTarget: [
                    D3D12_RENDER_TARGET_BLEND_DESC {
                        BlendEnable: false.into(),
                        LogicOpEnable: false.into(),
                        SrcBlend: D3D12_BLEND_ONE,
                        DestBlend: D3D12_BLEND_ZERO,
                        BlendOp: D3D12_BLEND_OP_ADD,
                        SrcBlendAlpha: D3D12_BLEND_ONE,
                        DestBlendAlpha: D3D12_BLEND_ZERO,
                        BlendOpAlpha: D3D12_BLEND_OP_ADD,
                        LogicOp: D3D12_LOGIC_OP_NOOP,
                        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    },
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                    D3D12_RENDER_TARGET_BLEND_DESC::default(),
                ],
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC::default(),
            SampleMask: u32::MAX,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

        unsafe { device.CreateGraphicsPipelineState(&desc) }
    }

    fn create_vertex_buffer(
        device: &ID3D12Device,
        aspect_ratio: f32,
    ) -> Result<(ID3D12Resource, D3D12_VERTEX_BUFFER_VIEW)> {
        let vertices = [
            Vertex {
                position: [0.0, 0.25 * aspect_ratio, 0.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            Vertex {
                position: [0.25, -0.25 * aspect_ratio, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
            Vertex {
                position: [-0.25, -0.25 * aspect_ratio, 0.0],
                color: [0.0, 0.0, 1.0, 1.0],
            },
        ];

        // Note: using upload heaps to transfer static data like vert buffers is
        // not recommended. Every time the GPU needs it, the upload heap will be
        // marshalled over. Please read up on Default Heap usage. An upload heap
        // is used here for code simplicity and because there are very few verts
        // to actually transfer.
        let mut vertex_buffer: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_UPLOAD,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: std::mem::size_of_val(&vertices) as u64,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut vertex_buffer,
            )?
        };
        let vertex_buffer = vertex_buffer.unwrap();

        // Copy the triangle data to the vertex buffer.
        unsafe {
            let mut data = std::ptr::null_mut();
            vertex_buffer.Map(0, None, Some(&mut data))?;
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), data as *mut Vertex, vertices.len());
            vertex_buffer.Unmap(0, None);
        }

        let vbv = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { vertex_buffer.GetGPUVirtualAddress() },
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
            SizeInBytes: std::mem::size_of_val(&vertices) as u32,
        };

        Ok((vertex_buffer, vbv))
    }

    #[repr(C)]
    struct Vertex {
        position: [f32; 3],
        color: [f32; 4],
    }

    fn get_current_flip_count(swap_chain: &IDXGISwapChain3) -> (u32, i64) {
        let mut present_stats: DXGI_FRAME_STATISTICS = DXGI_FRAME_STATISTICS::default();
        unsafe { swap_chain.GetFrameStatistics(&mut present_stats) };
        (present_stats.SyncRefreshCount, present_stats.SyncQPCTime)
    }

    fn wait_for_previous_frame(resources: &mut Resources) {
        // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST
        // PRACTICE. This is code implemented as such for simplicity. The
        // D3D12HelloFrameBuffering sample illustrates how to use fences for
        // efficient resource usage and to maximize GPU utilization.

        // Signal and increment the fence value.
        let fence = resources.fence_value;

        unsafe { resources.command_queue.Signal(&resources.fence, fence) }
            .ok()
            .unwrap();

        resources.fence_value += 1;

        // Wait until the previous frame is finished.
        if unsafe { resources.fence.GetCompletedValue() } < fence {
            unsafe {
                resources
                    .fence
                    .SetEventOnCompletion(fence, resources.fence_event)
            }
            .ok()
            .unwrap();

            unsafe { WaitForSingleObject(resources.fence_event, INFINITE) };
        }

        // get current scanline
        let mut scanline: D3DKMT_GETSCANLINE = Default::default();
        scanline.hAdapter = resources.wait_for_vblank_event.hAdapter;
        scanline.VidPnSourceId = resources.wait_for_vblank_event.VidPnSourceId;

        unsafe { D3DKMTGetScanLine(&mut scanline) };

        let swap_chain = &resources.swap_chain;
        let output: IDXGIOutput = unsafe { swap_chain.GetContainingOutput() }.unwrap();

        let count_before = *LAST_FLIP.lock().unwrap();

        let mut qpc_frequency = i64::default();
        unsafe { QueryPerformanceFrequency(&mut qpc_frequency) };

        // while scanline.InVerticalBlank.as_bool() == false {
        //     unsafe { D3DKMTGetScanLine(&mut scanline) };
        // }

        // while scanline.InVerticalBlank.as_bool() == true {
        //     unsafe { D3DKMTGetScanLine(&mut scanline) };
        // }

        //unsafe { output.WaitForVBlank().unwrap() };

        //unsafe { D3DKMTWaitForVerticalBlankEvent(&mut resources.wait_for_vblank_event) };

        // take timestamp
        let mut vblank_timestamp_wait = i64::default();
        unsafe { QueryPerformanceCounter(&mut vblank_timestamp_wait) };

        let (mut count_after, mut vblanc_timestamp_interupt) = get_current_flip_count(swap_chain);

        // busy wait until the flip count changes
        while count_after == count_before {
            (count_after, vblanc_timestamp_interupt) = get_current_flip_count(swap_chain);
        }

        let mut later = i64::default();
        unsafe { QueryPerformanceCounter(&mut later) };

        // measure the time from return from WaitForVBlank to the flip to be reported through the frame statistics
        let report_delay1 = (vblanc_timestamp_interupt - vblank_timestamp_wait) as f64
            / qpc_frequency as f64
            * 1_000_000.0;
        let report_delay2 =
            (later - vblank_timestamp_wait) as f64 / qpc_frequency as f64 * 1_000_000.0;
        // log::info!("Scanline: {}", scanline.ScanLine);
        // log::info!(
        //     "Report delay between WaitForVBlank and flip timestamped through frame statistics: {} us",
        //     report_delay1
        // );

        // log::info!(
        //     "Report delay between WaitForVBlank and flip polled through frame statistics: {} us",
        //     report_delay2
        // );

        // add the timestamp to frame_rate_calc
        let last_vblank_ms = vblanc_timestamp_interupt as f64 / qpc_frequency as f64 * 1000.0;
        //resources.frame_rate_calc.count_cycle(last_vblank_ms);

        // get current estimated fps
        //let fps = resources.frame_rate_calc.get_current_frequency();
        //log::info!("Current estimated FPS: {}", fps);

        let diff = count_after - count_before;

        log::info!("Flip count before {} after {}", count_before, count_after);

        // check if we missed any flips
        if diff > 1 {
            log::info!("Missed {} flips", diff - 1);
        } else if diff == 1 {
            //log::info!("No missed flips");
        } else {
            log::info!("Skipped {} flips", diff);
        }

        // add the flip count to the frame_times vector
        let vblank_to_vblank_interupt =
            (vblanc_timestamp_interupt - *LAST_INTERUPT_TIMESTAMP.lock().unwrap()) as f64
                / qpc_frequency as f64
                * 1000.0;

        let vblank_to_vblank_vblank = (vblank_timestamp_wait
            - *LAST_VLBANK_TIMESTAMP.lock().unwrap()) as f64
            / qpc_frequency as f64
            * 1000.0;

        resources
            .frame_times_interupt
            .push(vblank_to_vblank_interupt);

        resources
            .frame_times_wait_for_vblank
            .push(vblank_to_vblank_vblank);

        if resources.frame_times_interupt.len() > 100 {
            resources.frame_times_interupt.remove(0);
        }

        if resources.frame_times_wait_for_vblank.len() > 100 {
            resources.frame_times_wait_for_vblank.remove(0);
        }

        // // calculate the variance of the frame times
        // report_stats(&resources.frame_times_interupt, "FrameStatistics");
        // report_stats(&resources.frame_times_wait_for_vblank, "WaitForVBlank");

        resources.frame_index = unsafe { resources.swap_chain.GetCurrentBackBufferIndex() };

        *LAST_INTERUPT_TIMESTAMP.lock().unwrap() = vblanc_timestamp_interupt;
        *LAST_VLBANK_TIMESTAMP.lock().unwrap() = vblank_timestamp_wait;
        *LAST_FLIP.lock().unwrap() = count_after;
    }
}

enum VBlankEvent {
    VBlankBegin,
    VBlankEnd,
}

fn main() -> Result<()> {
    // set log level to info
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .unwrap();

    // create a channel to communicate with the scanline polling thread
    // let (tx, rx) = std::sync::mpsc::channel();

    // // create a thread to poll the scanline
    // std::thread::spawn(move || {
    //     assert!(set_current_thread_priority(ThreadPriority::Max).is_ok());

    //     let hndl = get_vblank_handle().unwrap();

    //     let mut scanline: D3DKMT_GETSCANLINE = Default::default();

    //     let mut calc = refresh_rates::RefreshRateCalculator::new();

    //     scanline.hAdapter = hndl.hAdapter;
    //     scanline.VidPnSourceId = hndl.VidPnSourceId;

    //     let mut was_in_vblank = false;

    //     let start = std::time::Instant::now();

    //     let mut vblank_time_vec = vec![start.elapsed().as_secs_f64()];

    //     let mut i = 0;

    //     loop {
    //         unsafe { D3DKMTGetScanLine(&mut scanline) };
    //         i += 1;

    //         let is_in_vblank = scanline.InVerticalBlank.as_bool();

    //         if !is_in_vblank && was_in_vblank {
    //             // We just left the vblank

    //             let t = start.elapsed().as_secs_f64() * 1000.0;
    //             vblank_time_vec.push(t);

    //             if vblank_time_vec.len() > 50 {
    //                 vblank_time_vec.remove(0);
    //             }

    //             // print stats
    //             let diffs: Vec<f64> = vblank_time_vec.windows(2).map(|w| w[1] - w[0]).collect();
    //             let last_diff = diffs.last().unwrap();
    //             //log::info!("Frame time: {} ms", last_diff);
    //             log::info!("Frame time: {} ms", last_diff);
    //             report_stats(&diffs, "VBlank");

    //             // // fps estimation
    //             // calc.count_cycle(t);
    //             // let fps = calc.get_current_frequency();
    //             // log::info!("Current estimated FPS: {}", fps);

    //             // we can sleep here to avoid busy waiting
    //             //std::thread::sleep(std::time::Duration::from_millis(1));

    //             i = 0;
    //         }

    //         was_in_vblank = is_in_vblank;
    //     }
    // });
    run_sample::<d3d12_hello_triangle::Sample>()?;
    Ok(())
}

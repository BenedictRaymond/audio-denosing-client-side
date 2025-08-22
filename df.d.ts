declare namespace wasm_bindgen {
	/* tslint:disable */
	/* eslint-disable */
	/**
	 * Create a DeepFilterNet Model
	 *
	 * Args:
	 *     - path: File path to a DeepFilterNet tar.gz onnx model
	 *     - atten_lim: Attenuation limit in dB.
	 *
	 * Returns:
	 *     - DF state doing the full processing: stft, DNN noise reduction, istft.
	 */
	export function df_create(model_bytes: Uint8Array, atten_lim: number): number;
	/**
	 * Get DeepFilterNet frame size in samples.
	 */
	export function df_get_frame_length(st: number): number;
	/**
	 * Set DeepFilterNet attenuation limit.
	 *
	 * Args:
	 *     - lim_db: New attenuation limit in dB.
	 */
	export function df_set_atten_lim(st: number, lim_db: number): void;
	/**
	 * Set DeepFilterNet post filter beta. A beta of 0 disables the post filter.
	 *
	 * Args:
	 *     - beta: Post filter attenuation. Suitable range between 0.05 and 0;
	 */
	export function df_set_post_filter_beta(st: number, beta: number): void;
	/**
	 * Processes a chunk of samples.
	 *
	 * Args:
	 *     - df_state: Created via df_create()
	 *     - input: Input buffer of length df_get_frame_length()
	 *     - output: Output buffer of length df_get_frame_length()
	 *
	 * Returns:
	 *     - Local SNR of the current frame.
	 */
	export function df_process_frame(st: number, input: Float32Array): Float32Array;
	export class DFState {
	  private constructor();
	  free(): void;
	}
	
}

declare type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

declare interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_dfstate_free: (a: number, b: number) => void;
  readonly df_create: (a: number, b: number, c: number) => number;
  readonly df_get_frame_length: (a: number) => number;
  readonly df_set_atten_lim: (a: number, b: number) => void;
  readonly df_set_post_filter_beta: (a: number, b: number) => void;
  readonly df_process_frame: (a: number, b: number, c: number) => any;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_start: () => void;
}

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
declare function wasm_bindgen (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

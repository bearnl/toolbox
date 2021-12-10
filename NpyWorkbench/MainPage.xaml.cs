using System;
using System.Runtime.InteropServices;
using System.Linq;
using Microsoft.Maui.Controls;
using NumSharp;


namespace NpyWorkbench
{
	public partial class MainPage : ContentPage
	{
		public MainPage()
		{
			InitializeComponent();
		}

		private async void OnOpenFileClicked(object sender, EventArgs e)
		{
			//var picker = new FileOpenPicker();
			//var initializeWithWindowWrapper = picker.As<IInitializeWithWindow>();
			//initializeWithWindowWrapper.Initialize(GetActiveWindow());
			//var file = await picker.PickSingleFileAsync();
			var path = "D:/Downloads/dataset/datasets/mycapture/mar-22-ranyi-dk-2-colour.npy";
			var matrix = np.load(path);
			ShapeLabel.Text = $"Matrix shape: [{String.Join(",", matrix.shape)}]";
			var firstImage = matrix[0];
			var preview = firstImage.ToBitmap();


		}

		//[ComImport, System.Runtime.InteropServices.Guid("3E68D4BD-7135-4D10-8018-9FB6D9F33FA1"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
		//public interface IInitializeWithWindow
		//{
		//	void Initialize([In] IntPtr hwnd);
		//}

		//[DllImport("user32.dll", ExactSpelling = true, CharSet = CharSet.Auto, PreserveSig = true, SetLastError = false)]
		//public static extern IntPtr GetActiveWindow();

		[DllImport("gdi32.dll", EntryPoint = "DeleteObject")]
		[return: MarshalAs(UnmanagedType.Bool)]
		public static extern bool DeleteObject([In] IntPtr hObject);

	}
}

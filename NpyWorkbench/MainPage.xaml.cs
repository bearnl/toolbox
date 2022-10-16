using Emgu.CV;
using Emgu.CV.Structure;
using Microsoft.UI.Xaml.Media.Imaging;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Backends;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Graphics.Imaging;
using System.ComponentModel;
using Windows.Web.UI;

namespace NpyWorkbench;

public partial class MainPage : ContentPage
{
	private InfoPageViewModel infoPageViewModel = new InfoPageViewModel();

	public MainPage()
	{
		InitializeComponent();
	}


	private async void OnOpenNpyClicked(object sender, EventArgs e)
	{
		var result = await FilePicker.Default.PickAsync(new PickOptions()
		{
			PickerTitle = "Select npy file to load",
			FileTypes = new FilePickerFileType(new Dictionary<DevicePlatform, IEnumerable<string>> {
				{ DevicePlatform.WinUI, new[] { ".npy" } }
			})
		});

		if (result == null)
			return;

		var dataset = np.load(result.FullPath);
		var frame = dataset[":1,:"].Clone();
		var bitmap = frame.ToBitmap();
		var stream = new MemoryStream();
		bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Bmp);
		stream.Position = 0;

		Preview.Source = ImageSource.FromStream(() => stream);

	}
}

public class InfoPageViewModel: INotifyPropertyChanged
{
	public event PropertyChangedEventHandler PropertyChanged;

	private string currentDataset;

	protected virtual void OnPropertyChanged(string propertyName)
	{
		PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
	}

	public string CurrentDataset
	{
		get { return currentDataset; }
		set
		{
			if (currentDataset != value)
			{
				currentDataset = value;
				OnPropertyChanged("CurrentDataset");
			}
		}
	}
}

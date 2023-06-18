using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;
namespace Face_Detection.Controllers
{
    [ApiController]
    [Route("api/v1/face")]
    public class FaceController : ControllerBase
    {
        private readonly CascadeClassifier _faceClassifier;
        private readonly CascadeClassifier _upperbodyclassifier;
        private readonly CascadeClassifier _fullbodyclassifier;
        public FaceController()
        {
            _faceClassifier = new CascadeClassifier("haarcascade_frontalface_default.xml");
            _upperbodyclassifier = new CascadeClassifier("haarcascade_upperbody.xml");
            _fullbodyclassifier = new CascadeClassifier("haarcascade_fullbody.xml");

        }
        [HttpPost("detect")]
        public IActionResult DetectFace([FromForm] IFormFile imageFile)
        {
            if (imageFile == null || imageFile.Length == 0)
            {
                return BadRequest("No file selected.");
            }

            using var memoryStream = new MemoryStream();
            imageFile.CopyTo(memoryStream);
            byte[] imageBytes = memoryStream.ToArray();
            using var mat = Cv2.ImDecode(imageBytes, ImreadModes.Color);
            using var grayMat = new Mat();
            Cv2.CvtColor(mat, grayMat, ColorConversionCodes.BGR2GRAY);
            var faces = _faceClassifier.DetectMultiScale(grayMat);
            var upperBodies = _upperbodyclassifier.DetectMultiScale(grayMat);
            var fullbody = _fullbodyclassifier.DetectMultiScale(grayMat);
            bool hasUpperBody = upperBodies.Length > 0;
            bool hasFullBody = fullbody.Length > 0;
            bool hasFace = faces.Length > 0;
            bool isCentered = false;
          /*  bool isPlainColor = false;*/
            if (hasFace)
            {
                var imageCenter = new Point2f(mat.Width / 2f, mat.Height / 2f);
                var faceCenter = new Point2f(faces[0].X + (faces[0].Width / 2f), faces[0].Y + (faces[0].Height / 2f));

                var tolerance = 0.5;

                isCentered = Math.Abs(imageCenter.X - faceCenter.X) <= tolerance * mat.Width &&
                             Math.Abs(imageCenter.Y - faceCenter.Y) <= tolerance * mat.Height;

               /* var backgroundRegion = new Rect(0, 0, mat.Width, mat.Height);
                var backgroundMat = new Mat(mat, backgroundRegion);

                var hist = new Mat();
                int[] channels = { 0 };
                int[] histSize = { 256 };
                Rangef[] ranges = { new Rangef(0, 256) };
                Cv2.CalcHist(new[] { backgroundMat }, channels, null, hist, channels.Length, histSize, ranges);

                Point minLoc, maxLoc;
                double minValue, maxValue;
                Cv2.MinMaxLoc(hist, out minValue, out maxValue, out minLoc, out maxLoc);
                var maxValIdx = maxLoc;

                var backgroundGrayValue = maxValIdx.Y;

                backgroundColor = new Scalar(backgroundGrayValue, backgroundGrayValue, backgroundGrayValue);

                isPlainColor = IsPlainColor(backgroundColor, 0.1);*/
            }

            var response = new
            {
                HasFace = hasFace,
                IsCentered = isCentered,
                HasUpperBody = hasUpperBody,
                HasFullBody = hasFullBody,
              /*  IsPlainColor = isPlainColor,*/
               /* BackgroundColor = new[] { backgroundColor.Val0, backgroundColor.Val1, backgroundColor.Val2 },*/
                ImageWidth = mat.Width,
                ImageHeight = mat.Height,
            
            };

            return Ok(response);
        }

        /*private bool IsPlainColor(Scalar color, double tolerance)
        {
            byte[] channels = { (byte)color.Val0, (byte)color.Val1, (byte)color.Val2 };

            int diff01 = Math.Abs(channels[0] - channels[1]);
            int diff02 = Math.Abs(channels[0] - channels[2]);
            int diff12 = Math.Abs(channels[1] - channels[2]);

            bool isPlainColor = diff01 <= tolerance && diff02 <= tolerance && diff12 <= tolerance &&
                               channels[0] >= tolerance && channels[1] >= tolerance && channels[2] >= tolerance;

            return isPlainColor;
        }*/
    }
}

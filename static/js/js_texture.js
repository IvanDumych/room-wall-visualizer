var input = document.getElementById('upload');
var infoArea = document.getElementById('upload-label');

input.addEventListener('change', showFileName);
function showFileName(event) {
    var input = event.srcElement;
    var fileName = input.files[0].name;
    infoArea.textContent = 'File name: ' + fileName;
}


window.addEventListener("DOMContentLoaded", (event) => {
    document.querySelector("#textureForm").addEventListener("submit", (e) => {
        e.preventDefault();

        var uploadForm = document.querySelector('input[type="file"]')
        const formData = new FormData();
        formData.append('file', uploadForm.files[0]);

        (async () => {
            const rawResponse = await fetch("/result_textured", {
                method: 'POST',
                body: formData
            });
            const content = await rawResponse.json();
            console.log(rawResponse.status);

            if (rawResponse.status == 200) {
                document.querySelector("#imageResult").src = content.room_path + "?t=" + new Date().getTime();
            }

            console.log(content);
        })();
    })

});
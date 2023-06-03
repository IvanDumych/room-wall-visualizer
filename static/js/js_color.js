window.addEventListener("DOMContentLoaded", (event) => {

    document.querySelector("#colorForm").addEventListener("submit", (e) => {
        e.preventDefault();
        let selectedColor = document.querySelector("#colorPicker").value;

        (async () => {
            const rawResponse = await fetch("/result_colored", {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ "color": selectedColor })
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


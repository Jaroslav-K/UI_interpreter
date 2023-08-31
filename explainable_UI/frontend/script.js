var features = null;
var featuresDom = [];

var selected = false;

var selected1 = null;
var selected2 = null;
var lastAsigned = false;

var checkbox1 = null;
var checkbox2 = null;

var myChart = null;

$(document).ready(function () {

    loadFeatures();
    getPrediction();

    initChart();


    function initChart() {
        console.log('Init chart.');
        const data = {
        datasets: [
          {
            label: 'Features',
            data: [{x: 0.5, y: 0.5}],
            backgroundColor: 'rgb(255, 99, 132)'
          }
        ]};

                        const config = {
                            type: 'scatter',
                            data: data,
                            options: {
                                elements:{
                                    point: {
                                        radius: 5,
                                        borderWidth: 3
                                    }
                                },
                                scales: {
                                    x: {
                                        type: 'linear',
                                        position: 'bottom',
                                        min: 0,
                                        max: 1,
                                        scaleLabel: {
                                            display: true,
                                            labelString: "Feature [x]"
                                        }
                                    },
                                    y: {
                                        type: 'linear',
                                        position: 'bottom',
                                        min: 0,
                                        max: 1,
                                        scaleLabel: {
                                            display: true,
                                            labelString: "Feature [y]"
                                        }
                                    }
                                },
                                maintainAspectRatio: false
                            }
                        };

        myChart = new Chart(
            document.getElementById('myChart'),
            config
        );
    }
});

function updateAll(){
  $.ajax({
      url: "http://127.0.0.1:5000/api",
      type: "post",
      dataType: "json",
      contentType: "application/json",
      data: JSON.stringify(features),
      success: function (response) {
          getServerData();

      },
      error: function (jqXHR, textStatus, errorThrown) {
          console.log('error');
      }
  });
}

function getServerData(){
  getPrediction();
  loadFeatures();
  setImage();
  setImageReconstructed();
}

$("#submit").click(function(e){
  e.preventDefault();
  console.log('clicked');
  uploadFile(allFiles[0]);
});

function uploadFile(file) {
  console.log('Uplaoding file');
  console.log('File='+file);
   var form = new FormData();
      form.append("file", file);
  $.ajax({
              url: 'http://127.0.0.1:5000/upload',
              type: 'POST',
              data: form,
                    processData: false,
                    contentType: false,
                    mimeType: "multipart/form-data",
              success:function(data){
                  console.log('Result=' + data);
                  clearSelectedFeature(1);
                  clearSelectedFeature(2);
                  getServerData();

              },
              cache: false
          });
}

function setImage(){
  $("#img-original").attr('src', 'http://127.0.0.1:5000/get_image?'+new Date().getTime());
}

function setImageReconstructed(){
    $("#img-reconstructed").attr('src', 'http://127.0.0.1:5000/get_image_reconstructed?'+new Date().getTime());
}

function getPrediction(){
  $.ajax({
      url: "http://127.0.0.1:5000/prediction",
      type: "GET",
      success: function (response) {
          console.log(response);
          updatePrediction(response);

      },
      error: function (jqXHR, textStatus, errorThrown) {
          console.log('error');
      }
  });
}

function loadFeatures() {
    $.ajax({
        url: 'http://127.0.0.1:5000/api',
        type: 'GET',
        success: function (data) {
          $("#features").empty();
          featuresDom = [];
          features = data;
          console.log('response=' + data.length);
          min = 0;
          max = 0;
          for(let i=0; i<data.length; i++){
            if(data[i].value < min) min=data[i].value;
            if(data[i].value > max) max=data[i].value;
          }
          console.log('min='+min + ' max='+max);
          for (let i = 0; i < data.length; i++) {
              console.log('selected1=' + selected1 + ' selected2=' + selected2);
              let checked = "";
              if(data[i].id == selected1 || data[i].id == selected2){
                console.log('checked data id=' + data[i].id);
                checked = "checked";
              }
              let html = '<div id="feature' + data[i].id + '" class="card m-1 p-1" style="max-width: 4rem; display:inline-block;"> \
                        <div class="card-header bg-transparent">' + data[i].id + '</div> \
                        <div class="card-body">\
                        <input disabled type="range" min="'+0+'" max="'+1+'" value="' + data[i].value + '" step="0.02" orient="vertical" onchange="change(this)" data-id="' + data[i].id + '"/>\
                        </div>\
                        <div class="card-footer bg-transparent" style="font-size: 8px">' + parseFloat(data[i].value).toFixed(2) + '</div> \
                        <div><input type="checkbox" data-id="' + data[i].id + '" onchange="featureSelected(this,'+ data[i].id + ')"'+ checked +'></div>\
                        </div>';
              let dom = $.parseHTML(html);
              featuresDom.push(dom);
              //console.log('html=' + html)
              $("#features").append(dom);
          }
        }
    });
}

function featureSelected(dom, id) {
    console.log('selected id=' + id + ' dom=' + dom.checked);
    if (dom.checked) {
        if (selected1 == null) {
            selected1 = id;
            lastAsigned = true;
            cloneSelectedFeature(1, id);
        } else if (selected2 == null) {
            selected2 = id;
            lastAsigned = false;
            cloneSelectedFeature(2, id);
        } else {
            //replace
            if (lastAsigned) {
              //uncheck
              uncheckFeature(selected2);
              selected2 = id;
              lastAsigned = false;

              cloneSelectedFeature(2, id);
            } else {
              //uncheck old
              uncheckFeature(selected1);

              selected1 = id;
              lastAsigned = true;

              cloneSelectedFeature(1, id);
            }
        }
    } else {
        if (selected1 == id) {
            selected1 = null;
            clearSelectedFeature(1);
        } else if (selected2 == id) {
            selected2 = null;
            clearSelectedFeature(2);
        } else {
            console.log('ERROR:[featureSelected]');
        }
    }
    console.log('selected1=' + selected1 + ' selected2=' + selected2);
}

function uncheckFeature(id){
  console.log('unchecking id='+id);
  for (let i = 0; i < featuresDom.length; i++) {
    let input = $(featuresDom[i]).find('input[type=checkbox]');
    if ($(input).data("id") == id) {
      console.log('unchecking!!!');
      $(input).prop('checked', false);
      return;
    }
  }
}

function cloneSelectedFeature(position, id) {
    clearSelectedFeature(position);
    let clone = document.querySelector('#feature' + id).cloneNode(true);
    clone.setAttribute('id', 'selectedFeature' + position);
    console.log('clone=' + clone);
    $(clone).find('input[type=checkbox]').remove();

    $(clone).find('input')[0].disabled = false;
    $("#selectedFeatures").append(clone);
}

function clearSelectedFeature(position) {
    if (position == 1) {
        $("#selectedFeatures").find('#selectedFeature1').remove();
    } else if (position == 2) {
        $("#selectedFeatures").find('#selectedFeature2').remove();
    }
}


function getIndex(id) {
    for (let i = 0; i < features.length; i++) {
        if (features[i].id == id)
            return i;
    }
    return -1;
}

 function updateChart(x, y){
        if(myChart != null){
            if(x != null)
                myChart.data.datasets[0].data[0].x = x;
            if(y != null)
                myChart.data.datasets[0].data[0].y = y;
            myChart.update();
        }
    }

//changing slider value
function change(element) {
    for (let i = 0; i < features.length; i++) {
        if (features[i].id == $(element).data("id")) {
            features[i].value = parseFloat($(element).val());
            console.log('id=' + features[i].id);
            let featureVal = $(element).parent().parent().find('.card-footer');
            $(featureVal).text($(element).val());

            updateSliders($(element).data("id"), $(element).val());
            console.log('val=' + featureVal);
            console.log('changing=' + $(element).data("id") + ' val=' + $(element).val());

            if(features[i].id == selected1){
                updateChart(features[i].value, null);
            }else if(features[i].id == selected2){
                updateChart(null, features[i].value);
            }
            updateAll();
            return;
        }
    }
}

function updateSliders(id, value) {
    console.log('updating slider');
    for (let i = 0; i < featuresDom.length; i++) {
        let input = $(featuresDom[i]).find('input[type=range]');
        console.log('input=' + input);
        if ($(input).data("id") == id) {
            $(input).val(value);
            let elem = $(featuresDom[i]).find('.card-footer');
            $(elem).text(value);
            return;
        }

    }
}
